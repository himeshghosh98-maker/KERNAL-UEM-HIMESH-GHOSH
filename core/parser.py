"""Chat parsers: WhatsApp .txt, JSON exports, and pasted-text transcripts."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.schema import Message

# WhatsApp export header patterns (multiple date/time locales).
WHATSAPP_PATTERNS: list[re.Pattern[str]] = [
    # [12/03/2026, 14:30:15] Alice: Hello
    re.compile(
        r"^\[(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+"
        r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APMapm]{2})?)\]\s+"
        r"(?P<sender>[^:]+?):\s+(?P<text>.*)$"
    ),
    # 12/03/2026, 14:30 - Alice: Hello
    re.compile(
        r"^(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+"
        r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APMapm]{2})?)\s*-\s+"
        r"(?P<sender>[^:]+?):\s+(?P<text>.*)$"
    ),
    # 2026-03-12 14:30:00 - Alice: Hello
    re.compile(
        r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s+"
        r"(?P<sender>[^:]+?):\s+(?P<text>.*)$"
    ),
]

SYSTEM_MARKERS = (
    "messages and calls are end-to-end encrypted",
    "created group", "added you", "changed the group description",
    "changed the subject", "left the group", "you were added",
    "security code changed", "turned on disappearing messages",
)


def _is_system_message(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in SYSTEM_MARKERS)


def _parse_dt(date_str: str, time_str: str) -> datetime:
    clean_date = date_str.strip().replace("-", "/")
    clean_time = time_str.strip().upper()
    candidate = f"{clean_date} {clean_time}"
    formats = [
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M",
        "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %I:%M:%S %p", "%m/%d/%y %I:%M %p",
        "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p",
        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    # Last resort: ISO date with bare time
    try:
        return datetime.fromisoformat(f"{clean_date}T{clean_time}")
    except ValueError:
        return datetime.now()


def parse_text(raw_text: str) -> list[Message]:
    """Parse WhatsApp-style or simple 'Sender: text' transcripts."""
    messages: list[Message] = []
    pending: dict[str, Any] | None = None
    pending_header = False  # True if pending started with a timestamped header
    next_id = 1

    def flush() -> None:
        nonlocal pending, pending_header
        if pending and not _is_system_message(pending["text"]):
            messages.append(
                Message(
                    id=pending["id"],
                    sender=pending["sender"],
                    timestamp=pending["timestamp"],
                    text=pending["text"].strip(),
                )
            )
        pending = None
        pending_header = False

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # System announcement: flush any pending message, skip the line
        if any(m in line.lower() for m in SYSTEM_MARKERS):
            flush()
            continue

        matched = False
        for pat in WHATSAPP_PATTERNS:
            m = pat.match(line)
            if m:
                flush()
                pending = {
                    "id": next_id,
                    "sender": m.group("sender").strip(),
                    "timestamp": _parse_dt(m.group("date"), m.group("time")),
                    "text": m.group("text").strip(),
                }
                pending_header = True
                next_id += 1
                matched = True
                break
        if matched:
            continue

        # "Sender: text" fallback — always starts a new message when there is
        # no timestamped header pending (multi-line WhatsApp bodies still fold in).
        parts = line.split(":", 1)
        simple_sender = (
            len(parts) == 2
            and 0 < len(parts[0].strip()) <= 35
            and not parts[0].strip().startswith("[")
        )
        if simple_sender and (pending is None or not pending_header):
            flush()
            pending = {
                "id": next_id,
                "sender": parts[0].strip(),
                "timestamp": datetime.now(),
                "text": parts[1].strip(),
            }
            pending_header = False
            next_id += 1
            continue

        if pending and pending_header:
            # Continuation of a multi-line WhatsApp message
            pending["text"] += "\n" + line
        elif pending:
            pending["text"] += "\n" + line
        else:
            pending = {
                "id": next_id,
                "sender": f"Participant {next_id}",
                "timestamp": datetime.now(),
                "text": line,
            }
            next_id += 1
    flush()
    return messages


def parse_json(raw: str | list | dict) -> list[Message]:
    """Parse a JSON conversation: a list of message objects (or {messages: [...]})."""
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    if isinstance(data, dict):
        data = data.get("messages", data.get("conversation", []))
    if not isinstance(data, list):
        raise ValueError("JSON conversation must be a list of message objects.")

    messages: list[Message] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        ts_raw = item.get("timestamp") or item.get("time") or item.get("date")
        if isinstance(ts_raw, (int, float)):
            dt = datetime.fromtimestamp(ts_raw)
        elif isinstance(ts_raw, str):
            try:
                dt = datetime.fromisoformat(ts_raw)
            except ValueError:
                dt = datetime.now()
        else:
            dt = datetime.now()
        reply_raw = item.get("reply_to") or item.get("replyTo")
        reply_to = int(reply_raw) if reply_raw is not None and str(reply_raw).isdigit() else None
        addressee = item.get("addressee") or item.get("target") or None
        messages.append(
            Message(
                id=int(item.get("id", i)),
                sender=str(item.get("sender") or item.get("author") or item.get("from") or "Unknown"),
                timestamp=dt,
                text=str(item.get("text") or item.get("message") or item.get("content") or ""),
                reply_to=reply_to,
                addressee=addressee,
            )
        )
    return messages


def parse_csv(raw: str) -> list[Message]:
    """Parse CSV with columns: id, sender, timestamp, text, reply_to, addressee."""
    messages: list[Message] = []
    reader = csv.DictReader(io.StringIO(raw))
    for i, row in enumerate(reader, start=1):
        sender = (row.get("sender") or row.get("name") or row.get("author") or "Unknown").strip()
        text = (row.get("text") or row.get("message") or row.get("content") or "").strip()
        ts_raw = (row.get("timestamp") or row.get("time") or row.get("date") or "").strip()
        try:
            dt = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now()
        except ValueError:
            dt = datetime.now()
        reply_raw = row.get("reply_to") or row.get("parent_id")
        reply_to = int(reply_raw) if reply_raw and str(reply_raw).isdigit() else None
        addressee = (row.get("addressee") or row.get("target") or "").strip() or None
        id_raw = row.get("id")
        try:
            msg_id = int(id_raw) if id_raw else i
        except ValueError:
            msg_id = i
        messages.append(Message(id=msg_id, sender=sender, timestamp=dt, text=text, reply_to=reply_to, addressee=addressee))
    return messages


def parse_any(raw_text: str) -> list[Message]:
    """Auto-detect format and parse. Returns [] on empty input."""
    stripped = raw_text.strip()
    if not stripped:
        return []
    if stripped[0] in "{[":
        try:
            return parse_json(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
    first_line = stripped.splitlines()[0].lower()
    if "," in first_line and any(k in first_line for k in ("sender", "timestamp", "text", "message")):
        try:
            return parse_csv(stripped)
        except csv.Error:
            pass
    return parse_text(stripped)


def parse_file(path: str | Path) -> list[Message]:
    """Parse a chat file by extension (.txt/.json/.csv)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    content = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json":
        return parse_json(content)
    if p.suffix.lower() == ".csv":
        return parse_csv(content)
    return parse_text(content)
