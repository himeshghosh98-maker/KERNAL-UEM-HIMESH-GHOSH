"""Optional Gemini verification of gap candidates.

Requires GEMINI_API_KEY. Strict JSON output from the model; every field is
validated. Any failure returns the input gaps unchanged — the app must work
fully with this flag OFF.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from core.config import DetectorConfig
from core.schema import Gap, Message

_PROMPT = """You are a conversation-audit assistant. Given detected communication
gaps with evidence, decide for each gap whether it is a REAL gap.

Respond with ONLY a JSON object (no markdown fences) of the form:
{{"verdicts": [{{"message_ids": [1,2], "keep": true, "reason": "..."}}]}}

Rules:
- keep=true only if the cited messages genuinely show the claimed gap.
- keep=false if the evidence actually resolves the gap or ids are wrong.
- Do not invent message ids.

Conversation excerpt:
{conversation}

Detected gaps:
{gaps}
"""


def _build_excerpt(messages: list[Message], max_msgs: int = 40) -> str:
    lines = []
    for m in messages[:max_msgs]:
        lines.append(f"[{m.id}] {m.sender}: {m.text}")
    return "\n".join(lines)


def _build_gaps_payload(gaps: list[Gap], max_gaps: int) -> str:
    payload = [
        {
            "gap_type": g.gap_type,
            "message_ids": g.message_ids,
            "explanation": g.explanation,
            "evidence": g.evidence,
        }
        for g in gaps[:max_gaps]
    ]
    return json.dumps(payload, ensure_ascii=False)


def _call_gemini(prompt: str, model: str, temperature: float) -> str:
    """Call Gemini via the REST API (google-generativeai optional; REST fallback)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Prefer official SDK if present
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(
            prompt,
            generation_config={"temperature": temperature, "response_mime_type": "application/json"},
        )
        return resp.text or ""
    except ImportError:
        pass

    # REST fallback
    import urllib.request

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — user-configured API
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {exc}") from exc


def _parse_strict_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from model output, tolerating code fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(text[start : end + 1])


def verify_gaps(
    gaps: list[Gap],
    messages: list[Message],
    config: DetectorConfig | None = None,
) -> list[Gap]:
    """Filter/keep gaps based on Gemini verdicts. Never raises to callers."""
    cfg = config or DetectorConfig()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not gaps:
        return gaps

    candidates = gaps[: cfg.gemini_max_gaps]
    prompt = _PROMPT.format(
        conversation=_build_excerpt(messages),
        gaps=_build_gaps_payload(candidates, cfg.gemini_max_gaps),
    )
    try:
        raw = _call_gemini(prompt, cfg.gemini_model, cfg.gemini_temperature)
        data = _parse_strict_json(raw)
        verdicts = data.get("verdicts", [])
        drop_keys: set[tuple] = set()
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            if v.get("keep") is False:
                ids = tuple(sorted(int(i) for i in v.get("message_ids", []) if str(i).isdigit()))
                if ids:
                    drop_keys.add(ids)
        kept = []
        for g in candidates:
            key = tuple(sorted(g.message_ids))
            if key in drop_keys:
                continue
            g.score_parts = {**g.score_parts, "llm_verified": 1.0}
            kept.append(g)
        # Gaps beyond the verify budget pass through unchanged
        return kept + gaps[len(candidates) :]
    except Exception as exc:  # noqa: BLE001 — verification is optional
        return gaps  # fail safe: keep original candidates
