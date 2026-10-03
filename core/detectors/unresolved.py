"""Detector: unresolved topics.

Segments the chat into topics via sliding-window shared content tokens +
TF-IDF cosine similarity. A topic is flagged as unresolved only when:
- it has at least `topic_min_msgs` messages,
- its body contains decision-relevant language (decide/should/option/...),
- the closing region has NO closure cue and NO social close ("thanks", "gn").
"""

from __future__ import annotations

import re

from core.detectors.base import BaseDetector
from core.schema import Gap, Message

_CONTENT_STOP = {
    "the", "and", "for", "you", "your", "are", "was", "were", "this", "that",
    "with", "have", "has", "had", "what", "when", "where", "who", "why",
    "how", "can", "could", "should", "would", "will", "please", "about",
    "from", "they", "them", "then", "than", "been", "being", "does", "did",
    "not", "but", "any", "all", "our", "out", "get", "got", "let", "just",
    "like", "know", "think", "want", "need", "make", "made", "come", "back",
    "some", "more", "most", "very", "much", "many", "here", "there", "into",
    "over", "under", "again", "still", "also", "even", "well", "way", "now",
    "yes", "yeah", "okay", "ok", "hi", "hey", "hello", "morning", "night",
}


class UnresolvedDetector(BaseDetector):
    name = "unresolved"
    gap_type = "unresolved"

    def detect(self, messages: list[Message]) -> list[Gap]:
        cfg = self.config
        if len(messages) < cfg.topic_min_msgs + 1:
            return []

        topic_of = self._segment_topics(messages)

        gaps: list[Gap] = []
        topics: dict[int, list[Message]] = {}
        for msg, tid in zip(messages, topic_of):
            topics.setdefault(tid, []).append(msg)

        for _tid, msgs in topics.items():
            if len(msgs) < cfg.topic_min_msgs:
                continue
            tail = msgs[-cfg.topic_last_msgs :]
            all_text = " ".join(m.text_lower for m in msgs)
            tail_text = " ".join(m.text_lower for m in tail)

            closure = any(cue in tail_text for cue in cfg.closure_cues)
            social_close = any(cue in tail_text for cue in cfg.social_close_cues)
            if closure or social_close:
                continue

            decision_like = any(lex in all_text for lex in cfg.decision_lexicon)
            has_question = any("?" in m.text for m in msgs)
            has_dispute = any(c in all_text for c in cfg.disagreement_cues)
            if not (decision_like or has_dispute):
                continue  # topic must be decision-relevant to count

            severity = 0.40
            severity += min(0.25, 0.03 * len(msgs))
            if any(cue in all_text for cue in cfg.deadline_cues):
                severity += 0.20
            if any(cue in all_text for cue in cfg.disagreement_cues):
                severity += 0.15
            severity = min(0.95, severity)

            ids = [m.id for m in msgs]
            evidence = [self._quote(m) for m in (msgs[0], msgs[-1])]
            gaps.append(
                Gap(
                    gap_type="unresolved",
                    severity=round(severity, 2),
                    message_ids=ids,
                    evidence=evidence,
                    explanation=(
                        f"Topic thread ({len(msgs)} messages) contains open decisions "
                        f"but ended without a closure cue or decision message."
                    ),
                    score_parts={
                        "base": 0.40,
                        "size": round(min(0.25, 0.03 * len(msgs)), 2),
                        "deadline": 0.20 if any(c in all_text for c in cfg.deadline_cues) else 0.0,
                        "disagreement": 0.15 if any(c in all_text for c in cfg.disagreement_cues) else 0.0,
                    },
                )
            )
        return gaps

    def _segment_topics(self, messages: list[Message]) -> list[int]:
        """Sliding-window topic assignment: significant shared tokens + TF-IDF."""
        cfg = self.config
        min_len = cfg.topic_min_tokens_len

        def content_tokens(text: str) -> set[str]:
            return {
                t
                for t in re.findall(r"[a-z0-9']{3,}", text.lower())
                if t not in _CONTENT_STOP and len(t) >= min_len
            }

        token_sets = [content_tokens(m.text) for m in messages]

        sims: list[list[float]] | None = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            texts = [m.text for m in messages]
            if any(t.strip() for t in texts):
                matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
                sims = cosine_similarity(matrix).tolist()
        except ValueError:
            sims = None

        topic_of: list[int] = [-1] * len(messages)
        topic_tokens: dict[int, set[str]] = {}
        next_tid = 0
        for i in range(len(messages)):
            if topic_of[i] != -1:
                continue
            tid = next_tid
            next_tid += 1
            topic_of[i] = tid
            topic_tokens[tid] = set(token_sets[i])
            for j in range(i + 1, min(i + 1 + cfg.topic_window, len(messages))):
                if topic_of[j] != -1:
                    continue
                overlap = len(token_sets[j] & topic_tokens[tid])
                sim = sims[i][j] if sims else 0.0
                if overlap >= cfg.topic_min_shared_tokens or sim >= cfg.topic_sim_threshold:
                    topic_of[j] = tid
                    topic_tokens[tid] |= token_sets[j]
        return topic_of
