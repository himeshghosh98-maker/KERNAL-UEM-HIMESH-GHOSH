from __future__ import annotations

from datetime import datetime

from core.addressee import annotate_addressees, infer_addressee, known_senders
from core.schema import Message


def _msg(id_: int, sender: str, text: str, reply_to: int | None = None) -> Message:
    return Message(id=id_, sender=sender, timestamp=datetime(2026, 3, 12, 10, id_), text=text, reply_to=reply_to)


def test_addressee_from_mention():
    msgs = [_msg(1, "Alice", "@Bob can you check this?"), _msg(2, "Bob", "sure")]
    out = annotate_addressees(msgs)
    assert out[0].addressee == "Bob"


def test_addressee_from_reply_to():
    msgs = [_msg(1, "Alice", "question for the team"), _msg(2, "Bob", "replying", reply_to=1)]
    out = annotate_addressees(msgs)
    assert out[1].addressee == "Alice"


def test_addressee_from_name_mention():
    msgs = [_msg(1, "Alice", "Charlie said the build is broken"), _msg(2, "Charlie", "yes it is")]
    out = annotate_addressees(msgs)
    assert out[0].addressee == "Charlie"


def test_addressee_defaults_to_previous_speaker_or_group():
    msgs = [_msg(1, "Alice", "hello all"), _msg(2, "Bob", "hey")]
    out = annotate_addressees(msgs)
    assert out[1].addressee == "Alice"  # previous speaker
    # first message, no mention -> group
    assert out[0].addressee in ("group", None)


def test_known_senders_order():
    msgs = [_msg(1, "Zoe", "a"), _msg(2, "Adam", "b"), _msg(3, "Zoe", "c")]
    assert known_senders(msgs) == ["Zoe", "Adam"]
