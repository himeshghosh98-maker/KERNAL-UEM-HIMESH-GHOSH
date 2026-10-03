"""Detector: unanswered questions and requests.

Finds questions / imperative requests with no reply from a *different* person
inside a configurable lookahead window (messages + time). A later message
counts as an answer if:
- it explicitly replies to the question, or
- it shares topical content words with the question (ignoring generic time words),
- or it starts with an answer cue that is substantive for that cue class.
"""

from __future__ import annotations

import re

from core.detectors.base import BaseDetector
from core.schema import Gap, Message

_WH_WORDS = {
    "who", "what", "where", "when", "why", "how", "which", "whose", "whom",
}
_REQUEST_PATTERNS = (
    re.compile(r"\bcan you\b", re.I),
    re.compile(r"\bcould you\b", re.I),
    re.compile(r"\bwould you\b", re.I),
    re.compile(r"\bwill you\b", re.I),
    re.compile(
        r"\bplease (send|share|check|review|update|confirm|provide|look|get|find|book|call|reply|grab)\b",
        re.I,
    ),
    re.compile(r"\bneeds? to\b", re.I),
    re.compile(r"\bkindly\b", re.I),
    re.compile(r"\bsomeone (please|needs? to|should)\b", re.I),
    re.compile(r"\bcan someone\b", re.I),
)
_WORD_RE = re.compile(r"[a-z0-9']{3,}")
_STOP = {
    "the", "and", "for", "you", "your", "are", "was", "were", "this", "that",
    "with", "have", "has", "had", "what", "when", "where", "who", "why",
    "how", "can", "could", "should", "would", "will", "please", "about",
    "from", "they", "them", "then", "than", "been", "being", "does", "did",
    "someone", "anyone", "everyone",
}

# Strong standalone cues vs cues that only count with topical overlap
_STANDALONE_CUES = {
    "done", "fixed", "sent", "here is", "here's", "already",
    "agreed", "sounds good", "working on it",
}
_POLAR_CUES = {"yes", "yeah", "yep", "no", "nope", "sure"}
_FILLER_PREFIXES = ("ok,", "okay,", "yeah,", "yep,", "hi,", "hey,", "sorry,", "thanks,", "thx,")


def _tokens(text: str, extra_stop: tuple[str, ...] = ()) -> set[str]:
    stop = _STOP | set(extra_stop)
    return {t for t in _WORD_RE.findall(text.lower()) if t not in stop}


def _is_question_or_request(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    lower = stripped.lower()
    first = re.split(r"\s+", lower)[0] if lower else ""
    if first in _WH_WORDS and len(stripped.split()) >= 3:
        return True
    return any(p.search(stripped) for p in _REQUEST_PATTERNS)


def _starts_with_cue(cl: str, cue: str) -> bool:
    """True if the message opens with the cue (optionally after filler words)."""
    probe = cl
    while True:
        probe2 = probe.lstrip()
        matched_filler = False
        for pref in _FILLER_PREFIXES:
            if probe2.startswith(pref):
                probe2 = probe2[len(pref) :].lstrip()
                matched_filler = True
                break
        if not matched_filler:
            break
    return probe2.startswith(cue)


class UnansweredDetector(BaseDetector):
    name = "unanswered"
    gap_type = "unanswered"

    def detect(self, messages: list[Message]) -> list[Gap]:
        cfg = self.config
        gaps: list[Gap] = []
        overlap_stop = tuple(cfg.overlap_stop)

        for i, msg in enumerate(messages):
            if not _is_question_or_request(msg.text):
                continue
            q_tokens = _tokens(msg.text, overlap_stop)
            has_request = any(p.search(msg.text) for p in _REQUEST_PATTERNS)
            # Tiny chatter ("lunch?") is skipped unless it is a real request/addressee
            if len(q_tokens) < 2 and not msg.addressee_explicit and not has_request:
                continue

            window = messages[i + 1 : i + 1 + cfg.unanswered_lookahead_messages]
            addressee_person = (
                msg.addressee
                if msg.addressee_explicit and msg.addressee and msg.addressee != "group"
                else None
            )
            answered = False
            for cand in window:
                if cand.sender == msg.sender:
                    continue  # own follow-ups are re-asks, not answers
                dt = (cand.timestamp - msg.timestamp).total_seconds() / 60.0
                if dt > cfg.unanswered_lookahead_minutes:
                    break
                # Explicit reply link
                if cand.reply_to == msg.id and len(_tokens(cand.text)) >= 1:
                    answered = True
                    break
                c_tokens = _tokens(cand.text, overlap_stop)
                topical = bool(q_tokens & c_tokens)
                cl = cand.text_lower.strip()
                from_addressee = addressee_person is not None and cand.sender == addressee_person

                # Topical content overlap. For explicit @person requests, a
                # stranger's topical remark is not an answer — the addressee
                # (or a clear completion cue) must engage.
                if topical and (addressee_person is None or from_addressee):
                    answered = True
                    break

                for cue in cfg.strong_answer_cues:
                    if not _starts_with_cue(cl, cue):
                        continue
                    if "?" in cand.text:
                        continue
                    # Polar cues (yes/no/sure) need topical overlap
                    if cue in _POLAR_CUES:
                        continue
                    # Completion cues: any sender, small amount of content
                    if cue in _STANDALONE_CUES:
                        if len(c_tokens) >= 2:
                            answered = True
                            break
                    # Volunteer / status cues: from anyone, some content
                    elif topical or len(c_tokens) >= 2:
                        answered = True
                        break
                if answered:
                    break

            if answered:
                continue

            urgent = bool(
                re.search(r"\b(urgent|asap|blocking|critical|deadline|today|tomorrow)\b", msg.text_lower)
            )
            target = msg.addressee if msg.addressee and msg.addressee != "group" else None
            severity = 0.45
            if target:
                severity += 0.20
            if urgent:
                severity += 0.20
            if not window:
                severity += 0.10
            severity = min(0.98, severity)

            who = f" addressed to @{target}" if target else ""
            gaps.append(
                Gap(
                    gap_type="unanswered",
                    severity=severity,
                    message_ids=[msg.id],
                    evidence=[self._quote(msg)],
                    explanation=(
                        f"{msg.sender}'s question/request{who} received no substantive reply "
                        f"from anyone else within the next {len(window)} messages."
                    ),
                    score_parts={
                        "base": 0.45,
                        "targeted": 0.20 if target else 0.0,
                        "urgent": 0.20 if urgent else 0.0,
                    },
                )
            )
        return gaps
