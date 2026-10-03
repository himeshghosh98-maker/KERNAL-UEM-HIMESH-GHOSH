from __future__ import annotations

from datetime import datetime

from core.parser import parse_any, parse_json, parse_text


WHATSAPP_SAMPLE = """\
[12/03/2026, 10:00:15] Alice: Hey everyone
[12/03/2026, 10:01:02] Bob: Morning!
Messages and calls are end-to-end encrypted.
[12/03/2026, 10:02:30] Alice: Can you finish the slides?
[12/03/2026, 10:03:10] Bob: multi-line
continuation here
"""

JSON_SAMPLE = """
[
  {"id": 1, "sender": "Alice", "timestamp": "2026-03-12T10:00:00", "text": "Hi", "reply_to": null, "addressee": null},
  {"id": 2, "sender": "Bob", "timestamp": "2026-03-12T10:01:00", "text": "Hey Alice", "reply_to": 1, "addressee": "Alice"}
]
"""


def test_parse_text_whatsapp():
    msgs = parse_text(WHATSAPP_SAMPLE)
    assert len(msgs) == 4  # system message filtered
    assert msgs[0].sender == "Alice"
    assert msgs[2].text == "Can you finish the slides?"
    assert "continuation" in msgs[3].text
    assert msgs[2].timestamp.year == 2026


def test_parse_text_simple_sender_colon():
    msgs = parse_text("Alice: hello\nBob: hi there")
    assert len(msgs) == 2
    assert msgs[1].sender == "Bob"


def test_parse_json():
    msgs = parse_json(JSON_SAMPLE)
    assert len(msgs) == 2
    assert msgs[1].reply_to == 1
    assert msgs[1].addressee == "Alice"
    assert isinstance(msgs[0].timestamp, datetime)


def test_parse_any_autodetect():
    assert len(parse_any(WHATSAPP_SAMPLE)) == 4
    assert len(parse_any(JSON_SAMPLE)) == 2
    assert parse_any("") == []
