"""Core data schema: Message and Gap."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Message:
    """A single conversation message.

    Attributes:
        id: Unique integer id (stable within one conversation).
        sender: Display name of the author.
        timestamp: When the message was sent.
        text: Raw message text.
        reply_to: Optional id of the message being replied to.
        addressee: Inferred intended recipient ("group", a person, or None).
    """

    id: int
    sender: str
    timestamp: datetime
    text: str
    reply_to: int | None = None
    addressee: str | None = None
    addressee_explicit: bool = False  # True if from @mention / reply_to / name mention

    @property
    def text_lower(self) -> str:
        return self.text.lower()


@dataclass
class Gap:
    """A detected communication gap with evidence.

    Attributes:
        gap_type: One of unanswered / ignored / clarification / unresolved.
        severity: 0.0..1.0 (higher = more serious).
        message_ids: Ids of the messages that constitute the gap.
        evidence: Exact quotes: "sender @ timestamp: text".
        explanation: One plain sentence why this is a gap.
        score_parts: Optional breakdown of severity contributions (for UI/docs).
    """

    gap_type: str
    severity: float
    message_ids: list[int]
    evidence: list[str]
    explanation: str
    score_parts: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gap_type": self.gap_type,
            "severity": round(self.severity, 3),
            "message_ids": list(self.message_ids),
            "evidence": list(self.evidence),
            "explanation": self.explanation,
            "score_parts": {k: round(v, 3) for k, v in self.score_parts.items()},
        }
