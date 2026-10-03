"""Addressee inference for multi-person chats.

Priority order:
1. Explicit reply_to -> the replied-to message's sender.
2. @mentions in the text (first mentioned known participant, else raw name).
3. Explicit addressee field from the parser (already set).
4. Name mentions (a known participant's name appearing in the text).
5. Else the previous speaker (reply-like continuation), or "group" if unknown.
"""

from __future__ import annotations

import re
from core.schema import Message

_MENTION_RE = re.compile(r"@([A-Za-z0-9_\-\.]+)")


def known_senders(messages: list[Message]) -> list[str]:
    """Unique sender names in order of first appearance."""
    seen: list[str] = []
    for m in messages:
        if m.sender not in seen:
            seen.append(m.sender)
    return seen


def _match_name(token: str, senders: list[str]) -> str | None:
    """Case-insensitive exact or first-name match against known senders."""
    t = token.lower().strip()
    for s in senders:
        if s.lower() == t:
            return s
    for s in senders:
        first = s.split()[0].lower()
        if first == t and len(t) >= 2:
            return s
    return None


def infer_addressee(
    msg: Message,
    messages: list[Message],
    senders: list[str] | None = None,
    by_id: dict[int, Message] | None = None,
) -> str | None:
    """Infer the intended addressee for a single message."""
    if senders is None:
        senders = known_senders(messages)
    if by_id is None:
        by_id = {m.id: m for m in messages}

    # 1. reply_to -> original sender
    if msg.reply_to is not None and msg.reply_to in by_id:
        return by_id[msg.reply_to].sender

    # 2. @mention
    for token in _MENTION_RE.findall(msg.text):
        matched = _match_name(token, senders)
        return matched or token

    # 3. Explicit field already populated by parser
    if msg.addressee and msg.addressee.lower() != "group":
        matched = _match_name(msg.addressee, senders)
        return matched or msg.addressee

    # 4. Bare name mentions (exclude self-mention)
    lower = msg.text_lower
    for s in senders:
        if s == msg.sender:
            continue
        # Word-boundary-ish match for full name or first name
        names = {s.lower()}
        parts = s.split()
        if len(parts) >= 2:
            names.add(parts[0].lower())
            names.add(parts[-1].lower())
        for name in names:
            if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lower):
                return s

    # 5. Previous speaker (continuation) or "group"
    idx = next((i for i, m in enumerate(messages) if m.id == msg.id), None)
    if idx is not None and idx > 0:
        prev = messages[idx - 1]
        if prev.sender != msg.sender:
            return prev.sender
    return "group"


def annotate_addressees(messages: list[Message]) -> list[Message]:
    """Return messages with addressee filled in (new list, same Message dataclass).

    addressee_explicit is True only when inference used a hard signal
    (@mention, reply_to, or a known participant name in the text) — the
    previous-speaker fallback is soft and must not trigger 'ignored' gaps.
    """
    senders = known_senders(messages)
    by_id = {m.id: m for m in messages}
    out: list[Message] = []
    for m in messages:
        addr, explicit = _infer_addressee_with_source(m, messages, senders=senders, by_id=by_id)
        out.append(
            Message(
                id=m.id,
                sender=m.sender,
                timestamp=m.timestamp,
                text=m.text,
                reply_to=m.reply_to,
                addressee=addr,
                addressee_explicit=explicit,
            )
        )
    return out


def _infer_addressee_with_source(
    msg: Message,
    messages: list[Message],
    senders: list[str],
    by_id: dict[int, Message],
) -> tuple[str | None, bool]:
    """Return (addressee, explicit_flag)."""
    # 1. reply_to -> original sender (explicit)
    if msg.reply_to is not None and msg.reply_to in by_id:
        return by_id[msg.reply_to].sender, True

    # 2. @mention (explicit)
    for token in _MENTION_RE.findall(msg.text):
        matched = _match_name(token, senders)
        return (matched or token), True

    # 3. Explicit field already populated by parser (explicit)
    if msg.addressee and msg.addressee.lower() != "group":
        matched = _match_name(msg.addressee, senders)
        return (matched or msg.addressee), True

    # 4. Bare name mentions (explicit) — exclude self-mention
    lower = msg.text_lower
    for s in senders:
        if s == msg.sender:
            continue
        names = {s.lower()}
        parts = s.split()
        if len(parts) >= 2:
            names.add(parts[0].lower())
            names.add(parts[-1].lower())
        for name in names:
            if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lower):
                return s, True

    # 5. Previous speaker (soft) or "group"
    idx = next((i for i, m in enumerate(messages) if m.id == msg.id), None)
    if idx is not None and idx > 0:
        prev = messages[idx - 1]
        if prev.sender != msg.sender:
            return prev.sender, False
    return "group", False
