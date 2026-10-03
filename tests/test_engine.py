from __future__ import annotations

import json
from datetime import datetime

from core.engine import EvidenceEngine
from core.schema import Gap, Message


def test_json_export_roundtrip():
    msgs = [
        Message(id=1, sender="A", timestamp=datetime(2026, 3, 12, 9, 0), text="Hello"),
        Message(id=2, sender="B", timestamp=datetime(2026, 3, 12, 9, 1), text="Hi"),
    ]
    result = EvidenceEngine().analyze(msgs)
    data = json.loads(result.to_json())
    assert data["total_messages"] == 2
    assert "gaps" in data
    assert data["participants"] == ["A", "B"]


def test_evidence_quotes_exact_text():
    msgs = [
        Message(id=1, sender="Alice", timestamp=datetime(2026, 3, 12, 10, 5), text="Can someone deploy this today?"),
        Message(id=2, sender="Bob", timestamp=datetime(2026, 3, 12, 10, 6), text="nice meme"),
    ]
    result = EvidenceEngine().analyze(msgs)
    blob = " ".join(" ".join(g.evidence) for g in result.gaps)
    assert "Can someone deploy this today?" in blob
    assert "Alice" in blob


def test_dedupe_overlapping_same_type():
    # Engine merges heavily overlapping same-type candidates.
    from core.config import DetectorConfig

    cfg = DetectorConfig()
    engine = EvidenceEngine(cfg)
    msgs = [
        Message(id=1, sender="A", timestamp=datetime(2026, 3, 12, 9, 0), text="what do you mean?"),
        Message(id=2, sender="A", timestamp=datetime(2026, 3, 12, 9, 5), text="what do you mean exactly?"),
    ]
    result = engine.analyze(msgs)
    clar = [g for g in result.gaps if g.gap_type == "clarification"]
    assert clar
    # ids unique within a gap
    for g in clar:
        assert len(g.message_ids) == len(set(g.message_ids))
