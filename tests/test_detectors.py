from __future__ import annotations

from datetime import datetime, timedelta

from core.detectors import (
    ClarificationDetector,
    IgnoredDetector,
    UnansweredDetector,
    UnresolvedDetector,
)
from core.engine import EvidenceEngine
from core.schema import Message


def _mk(id_: int, sender: str, text: str, minutes: int = 0, reply_to: int | None = None) -> Message:
    return Message(
        id=id_,
        sender=sender,
        timestamp=datetime(2026, 3, 12, 9, 0) + timedelta(minutes=minutes),
        text=text,
        reply_to=reply_to,
    )


def test_unanswered_question_flagged():
    msgs = [
        _mk(1, "Alice", "Can you finish the slides by 5?"),
        _mk(2, "Bob", "lol nice weather today"),
        _mk(3, "Carol", "yeah sunny"),
    ]
    gaps = UnansweredDetector().detect(msgs)
    assert any(g.gap_type == "unanswered" for g in gaps)
    assert 1 in gaps[0].message_ids


def test_unanswered_question_answered_not_flagged():
    msgs = [
        _mk(1, "Alice", "Can you finish the slides by 5?"),
        _mk(2, "Bob", "yes I will send the slides soon"),
    ]
    gaps = UnansweredDetector().detect(msgs)
    assert not any(g.gap_type == "unanswered" and 1 in g.message_ids for g in gaps)


def test_ignored_direct_address():
    from core.addressee import annotate_addressees

    msgs = annotate_addressees(
        [
            _mk(1, "Alice", "@Bob please review the API contract today"),
            _mk(2, "Carol", "lunch anyone?"),
            _mk(3, "Alice", "moving on"),
            _mk(4, "Carol", "sure"),
            _mk(5, "Alice", "anyway"),
        ]
    )
    gaps = IgnoredDetector().detect(msgs)
    assert any(g.gap_type == "ignored" and 1 in g.message_ids for g in gaps)


def test_clarification_cue():
    msgs = [_mk(1, "Bob", "what do you mean by \"migration\"?"), _mk(2, "Alice", "ok")]
    gaps = ClarificationDetector().detect(msgs)
    assert any(g.gap_type == "clarification" for g in gaps)


def test_clarification_reask_similarity():
    msgs = [
        _mk(1, "Bob", "what time is the deployment happening on Friday?"),
        _mk(2, "Alice", "not sure yet"),
        _mk(3, "Bob", "what time is the deployment happening on friday then?"),
    ]
    gaps = ClarificationDetector().detect(msgs)
    assert any(
        g.gap_type == "clarification" and set(g.message_ids) >= {1, 3} for g in gaps
    )


def test_unresolved_topic_without_closure():
    msgs = [
        _mk(1, "Alice", "we need to decide on the venue for the event"),
        _mk(2, "Bob", "the rooftop venue looks good for the event"),
        _mk(3, "Carol", "but budget for the event is tight"),
        _mk(4, "Alice", "the event catering is still open"),
        _mk(5, "Bob", "event decorations could be cheaper"),
    ]
    gaps = UnresolvedDetector().detect(msgs)
    assert any(g.gap_type == "unresolved" for g in gaps)


def test_resolved_topic_not_flagged():
    msgs = [
        _mk(1, "Alice", "should we use Redis or Memcached for cache"),
        _mk(2, "Bob", "Redis is better for persistence"),
        _mk(3, "Alice", "agreed, let's go with Redis, done"),
    ]
    gaps = UnresolvedDetector().detect(msgs)
    assert not any(g.gap_type == "unresolved" for g in gaps)


def test_engine_rejects_invalid_ids_and_ranks():
    msgs = [
        _mk(1, "Alice", "@Bob can you check the server logs today?"),
        _mk(2, "Carol", "coffee break"),
        _mk(3, "Alice", "bob?"),
    ]
    engine = EvidenceEngine()
    result = engine.analyze(msgs)
    assert result.gaps, "expected at least one gap"
    valid_ids = {m.id for m in result.messages}
    for g in result.gaps:
        assert set(g.message_ids) <= valid_ids
        assert g.evidence
        assert 0.0 <= g.severity <= 1.0
    # ranked descending
    sev = [g.severity for g in result.gaps]
    assert sev == sorted(sev, reverse=True)
