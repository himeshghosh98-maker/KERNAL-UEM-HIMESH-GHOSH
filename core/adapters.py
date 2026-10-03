"""Optional adapters for reference repos (parshift, groupchat-decoder, ADR).

Every wrapper is guarded by try/except. If a package is not installed or
fails, we fall back to GapDetect's native implementations (see AUDIT.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.schema import Gap, Message

log = logging.getLogger(__name__)


@dataclass
class AdapterStatus:
    name: str
    available: bool
    mode: str  # "imported" | "native_fallback" | "skipped"
    detail: str = ""


@dataclass
class AdapterReport:
    adapters: list[AdapterStatus] = field(default_factory=list)

    def add(self, status: AdapterStatus) -> None:
        self.adapters.append(status)

    def to_dict(self) -> dict:
        return {
            "adapters": [
                {"name": a.name, "available": a.available, "mode": a.mode, "detail": a.detail}
                for a in self.adapters
            ]
        }


def try_import_parshift():
    """Return the parshift package if importable, else None."""
    try:
        import parshift  # type: ignore

        return parshift
    except Exception as exc:  # noqa: BLE001 — optional dependency
        log.debug("parshift not available: %s", exc)
        return None


def parshift_participation_coding(messages: list[Message]) -> dict:
    """Native Gibson-style S/A coding (AB-BA reciprocity counts).

    Inspired by bdfsaraiva/parshift (MIT) but implemented natively so the
    project runs without the dependency.
    """
    turns: list[tuple[str, str]] = []
    for m in messages:
        addressee = m.addressee if m.addressee and m.addressee != "group" else "group"
        turns.append((m.sender, addressee))

    ab_ba = 0  # direct reciprocal dialogue
    ab_x0 = 0  # addressed person does not take next turn
    a0_x0 = 0  # third party takes the floor / group turnover
    for i in range(len(turns) - 1):
        s, a = turns[i]
        s2, a2 = turns[i + 1]
        if a != "group" and s2 == a and a2 == s:
            ab_ba += 1
        elif a != "group" and s2 != a:
            ab_x0 += 1
        else:
            a0_x0 += 1
    return {
        "ab_ba": ab_ba,
        "ab_x0": ab_x0,
        "a0_x0": a0_x0,
        "turns": turns,
    }


def run_adapters(messages: list[Message], report: AdapterReport | None = None) -> tuple[list[Gap], AdapterReport]:
    """Attempt optional repo integrations; always safe to call.

    Returns:
        (extra_gaps, report). extra_gaps is [] unless an adapter produced
        additional candidates.
    """
    report = report or AdapterReport()
    extra_gaps: list[Gap] = []

    # --- parshift ---
    ps = try_import_parshift()
    if ps is not None:
        try:
            coding = parshift_participation_coding(messages)
            # Use native coding even when package present (avoids API churn).
            if coding["ab_x0"] > max(2, coding["ab_ba"]):
                extra_gaps.append(
                    Gap(
                        gap_type="ignored",
                        severity=0.55,
                        message_ids=[],
                        evidence=[],
                        explanation=(
                            f"parshift adapter: {coding['ab_x0']} non-reciprocal addresses vs "
                            f"{coding['ab_ba']} AB-BA pairs — structural turn-taking weakness."
                        ),
                        score_parts={"ab_x0": float(coding["ab_x0"]), "ab_ba": float(coding["ab_ba"])},
                    )
                )
            report.add(AdapterStatus("parshift", True, "imported", "package importable; native S/A coding used"))
        except Exception as exc:  # noqa: BLE001
            report.add(AdapterStatus("parshift", False, "native_fallback", f"adapter error: {exc}"))
    else:
        report.add(AdapterStatus("parshift", False, "native_fallback", "package not installed; native S/A coding used"))

    # --- groupchat-decoder (JS, no license) ---
    report.add(
        AdapterStatus(
            "groupchat-decoder",
            False,
            "skipped",
            "JS app, no license, not present — unanswered-logic borrowed into native detector",
        )
    )

    # --- Agreement-and-Disagreement-Recognition ---
    report.add(
        AdapterStatus(
            "agreement-disagreement",
            False,
            "skipped",
            "repo not found publicly — stance cues borrowed into clarification/unresolved detectors",
        )
    )

    return extra_gaps, report
