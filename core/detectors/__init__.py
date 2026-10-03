"""Detector package: one module per gap type, all subclassing BaseDetector."""

from core.detectors.base import BaseDetector
from core.detectors.unanswered import UnansweredDetector
from core.detectors.ignored import IgnoredDetector
from core.detectors.clarification import ClarificationDetector
from core.detectors.unresolved import UnresolvedDetector

__all__ = [
    "BaseDetector",
    "UnansweredDetector",
    "IgnoredDetector",
    "ClarificationDetector",
    "UnresolvedDetector",
]


def all_detectors(config=None):
    """Instantiate the four built-in detectors with a shared config."""
    from core.config import DEFAULT_CONFIG

    cfg = config or DEFAULT_CONFIG
    return [
        UnansweredDetector(cfg),
        IgnoredDetector(cfg),
        ClarificationDetector(cfg),
        UnresolvedDetector(cfg),
    ]
