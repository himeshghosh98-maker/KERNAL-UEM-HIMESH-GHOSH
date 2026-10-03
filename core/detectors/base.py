"""Base class for all gap detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.config import DEFAULT_CONFIG, DetectorConfig
from core.schema import Gap, Message


class BaseDetector(ABC):
    """Interface every detector must implement."""

    name: str = "base"
    gap_type: str = "unknown"

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    @abstractmethod
    def detect(self, messages: list[Message]) -> list[Gap]:
        """Return candidate Gaps (evidence/severity may be refined by the engine)."""

    def _quote(self, msg: Message) -> str:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M") if msg.timestamp else "?"
        return f"{msg.sender} @ {ts}: {msg.text}"
