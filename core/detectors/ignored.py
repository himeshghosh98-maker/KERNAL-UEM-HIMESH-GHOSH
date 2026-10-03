"""Detector: ignored / unanswered direct addresses.

Only *explicit* addressees count (@mention, reply_to, or a named participant in
the text). The previous-speaker fallback is soft and must not create false
"ignored" gaps in casual group chat. A gap exists when that addressee sends
nothing within the lookahead window.
"""

from __future__ import annotations

import re

from core.detectors.base import BaseDetector
from core.schema import Gap, Message

_DIRECTED_SHAPE = re.compile(
    r"(@\w+|\?|can you|could you|please|kindly|send|share|review|check|confirm|help me|get )",
    re.I,
)
_WEAK_TOKENS = {
    "the", "and", "for", "you", "your", "are", "was", "were", "this", "that",
    "with", "have", "has", "had", "what", "when", "where", "who", "why",
    "how", "can", "could", "should", "would", "will", "please", "about",
    "from", "they", "them", "then", "than", "been", "being", "does", "did",
    "not", "but", "any", "all", "our", "out", "get", "got", "let", "just",
    "like", "know", "think", "want", "need", "make", "made", "come", "back",
    "some", "more", "most", "very", "much", "many", "here", "there", "into",
    "over", "under", "again", "still", "also", "even", "well", "way", "now",
}
_COMPLETION_CUES = (
    "done", "fixed", "sent", "here is", "here's", "already",
    "agreed", "sounds good", "confirmed", "uploaded", "booked", "shared",
)


class IgnoredDetector(BaseDetector):
    name = "ignored"
    gap_type = "ignored"

    def detect(self, messages: list[Message]) -> list[Gap]:
        cfg = self.config
        gaps: list[Gap] = []

        for i, msg in enumerate(messages):
            addressee = msg.addressee
            if not addressee or addressee.lower() == "group":
                continue
            if not getattr(msg, "addressee_explicit", False):
                continue  # soft previous-speaker inference is not a direct address
            if len(msg.text.split()) < cfg.ignored_min_words:
                continue
            if not _DIRECTED_SHAPE.search(msg.text):
                continue  # casual chatter is not a direct address worth flagging

            window = messages[i + 1 : i + 1 + cfg.ignored_lookahead_messages]
            # Need enough following messages for a fair chance to respond
            if len(window) < cfg.ignored_min_window:
                continue

            addressee_msgs = [cand for cand in window if cand.sender == addressee]
            addressee_spoke = bool(addressee_msgs)
            time_ok = any(
                (cand.timestamp - msg.timestamp).total_seconds() / 60.0 <= cfg.ignored_lookahead_minutes
                for cand in window
            )
            if not time_ok:
                continue

            fulfilled = False
            if addressee_spoke:
                # Direct address is "handled" only if the addressee's replies
                # actually engage the request (content overlap or completion cue).
                req_tokens = {
                    t
                    for t in re.findall(r"[a-z0-9']{3,}", msg.text_lower)
                    if t not in _WEAK_TOKENS
                }
                for cand in addressee_msgs:
                    c_tokens = {
                        t
                        for t in re.findall(r"[a-z0-9']{3,}", cand.text_lower)
                        if t not in _WEAK_TOKENS
                    }
                    if req_tokens & c_tokens:
                        fulfilled = True
                        break
                    cl = cand.text_lower.strip()
                    if any(
                        cl.startswith(cue) or cl[len(cl.split(None, 1)[0]) :].strip().startswith(cue)
                        for cue in _COMPLETION_CUES
                    ):
                        if len(c_tokens) >= 2:
                            fulfilled = True
                            break
                if fulfilled:
                    continue
            # Silent addressee OR only non-engaging replies => ignored

            people_involved = len({msg.sender, addressee})
            severity = 0.50
            if msg.reply_to is not None:
                severity += 0.15
            if len(msg.text.split()) >= 8:
                severity += 0.10
            severity = min(0.95, severity)

            evidence = [self._quote(msg)]
            explanation = (
                f"{msg.sender} addressed @{addressee}, who sent nothing within the next "
                f"{len(window)} messages — the direct address was ignored."
            )
            if msg.reply_to is not None:
                explanation = (
                    f"@{addressee} never responded to a reply from {msg.sender} "
                    f"(thread on message {msg.reply_to}) within {len(window)} messages."
                )

            gaps.append(
                Gap(
                    gap_type="ignored",
                    severity=severity,
                    message_ids=[msg.id],
                    evidence=evidence,
                    explanation=explanation,
                    score_parts={
                        "base": 0.50,
                        "thread": 0.15 if msg.reply_to is not None else 0.0,
                        "detailed": 0.10 if len(msg.text.split()) >= 8 else 0.0,
                        "people": float(people_involved),
                    },
                )
            )
        return gaps
