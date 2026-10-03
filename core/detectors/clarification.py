"""Detector: repeated clarification requests.

Two signals:
1. Explicit clarification cue phrases ("what do you mean", "can you clarify"...).
2. The *same person* re-asking a similar question (TF-IDF cosine similarity
   above threshold, or optional sentence-transformers embeddings).
"""

from __future__ import annotations

import re

from core.detectors.base import BaseDetector
from core.schema import Gap, Message


def _tfidf_sim(a: str, b: str) -> float:
    """Lightweight cosine similarity over bag-of-words with IDF weighting."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not a.strip() or not b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
    except ValueError:
        return 0.0


class ClarificationDetector(BaseDetector):
    name = "clarification"
    gap_type = "clarification"

    def detect(self, messages: list[Message]) -> list[Gap]:
        cfg = self.config
        gaps: list[Gap] = []
        cues = tuple(cfg.clarification_cues)

        # --- Signal 1: explicit clarification cues ---
        for i, msg in enumerate(messages):
            lower = msg.text_lower
            hit = next((c for c in cues if c in lower), None)
            if not hit:
                continue
            # Severity grows if the same person keeps asking
            reasks = sum(
                1
                for m in messages[i + 1 : i + 1 + cfg.clarification_reask_window]
                if m.sender == msg.sender
                and any(c in m.text_lower for c in cues)
            )
            severity = 0.40 + min(0.30, 0.10 * reasks)
            if reasks > 0:
                severity = min(0.90, severity + 0.15)
            gaps.append(
                Gap(
                    gap_type="clarification",
                    severity=round(min(0.95, severity), 2),
                    message_ids=[msg.id],
                    evidence=[self._quote(msg)],
                    explanation=(
                        f"{msg.sender} asked for clarification (\"{hit}\")"
                        + (f" and repeated the request {reasks} more time(s)." if reasks else ".")
                    ),
                    score_parts={"base": 0.40, "reasks": round(0.10 * reasks, 2)},
                )
            )

        # --- Signal 2: same person re-asking a similar question ---
        for i, msg in enumerate(messages):
            if "?" not in msg.text:
                continue
            lookback = messages[max(0, i - cfg.clarification_reask_window) : i]
            for prev in reversed(lookback):
                if prev.sender != msg.sender:
                    continue
                if "?" not in prev.text:
                    continue
                sim = _tfidf_sim(prev.text, msg.text)
                if sim >= cfg.clarification_similarity_threshold:
                    # Avoid double-counting if a cue-phrase gap already covers this pair
                    already = any(
                        prev.id in g.message_ids and msg.id in g.message_ids for g in gaps
                    )
                    if already:
                        continue
                    severity = min(0.90, 0.45 + 0.45 * sim)
                    gaps.append(
                        Gap(
                            gap_type="clarification",
                            severity=round(severity, 2),
                            message_ids=[prev.id, msg.id],
                            evidence=[self._quote(prev), self._quote(msg)],
                            explanation=(
                                f"{msg.sender} re-asked a similar question (cosine sim {sim:.2f}) "
                                f"after the earlier question at message {prev.id} — unclear first time."
                            ),
                            score_parts={"base": 0.45, "similarity": round(0.45 * sim, 2)},
                        )
                    )
                    break
        return gaps
