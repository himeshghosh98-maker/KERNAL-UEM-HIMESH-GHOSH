"""Evidence engine — merge, validate, score, and rank gap candidates.

Guarantees:
- Every Gap cites real message ids from the conversation (invalid ids rejected).
- evidence = exact quotes with sender and timestamp.
- severity in 0..1 from time unanswered, re-asks, people involved, and
  deadline/decision cue words.
- Output: ranked list of Gap objects + JSON export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from core.adapters import AdapterReport, run_adapters
from core.addressee import annotate_addressees
from core.config import DEFAULT_CONFIG, DetectorConfig
from core.detectors import all_detectors
from core.schema import Gap, Message


@dataclass
class AnalysisResult:
    messages: list[Message]
    gaps: list[Gap]
    participants: list[str] = field(default_factory=list)
    adapter_report: AdapterReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def gaps_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for g in self.gaps:
            counts[g.gap_type] = counts.get(g.gap_type, 0) + 1
        return counts

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_messages": len(self.messages),
            "participants": self.participants,
            "gaps_by_type": self.gaps_by_type(),
            "gaps": [g.to_dict() for g in self.gaps],
            "adapters": self.adapter_report.to_dict() if self.adapter_report else {},
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, ensure_ascii=False)


class EvidenceEngine:
    """Runs detectors + adapters, dedupes candidates, validates ids, scores gaps."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def analyze(
        self,
        messages: list[Message],
        use_adapters: bool = True,
        use_embeddings: bool = False,
        use_llm_verify: bool = False,
    ) -> AnalysisResult:
        cfg = self.config
        messages = annotate_addressees(messages)
        by_id = {m.id: m for m in messages}
        participants = sorted({m.sender for m in messages})

        candidates: list[Gap] = []
        for det in all_detectors(cfg):
            try:
                candidates.extend(det.detect(messages))
            except Exception as exc:  # noqa: BLE001 — detectors must not crash the app
                candidates.append(
                    Gap(
                        gap_type="error",
                        severity=0.0,
                        message_ids=[],
                        evidence=[],
                        explanation=f"Detector {det.name} failed: {exc}",
                    )
                )

        adapter_report: AdapterReport | None = None
        if use_adapters:
            extra, adapter_report = run_adapters(messages)
            candidates.extend(extra)

        if use_embeddings:
            candidates = self._optional_embedded_clarification(messages, candidates)

        gaps = self._merge_and_validate(candidates, by_id)
        gaps = self._score(gaps, by_id, messages)
        gaps = [g for g in gaps if g.severity >= cfg.min_severity and g.gap_type != "error"]
        gaps.sort(key=lambda g: (-g.severity, g.gap_type, g.message_ids[0] if g.message_ids else 0))

        if use_llm_verify:
            try:
                from core.llm_verify import verify_gaps

                gaps = verify_gaps(gaps, messages, cfg)
            except Exception as exc:  # noqa: BLE001
                self.metadata_note = f"LLM verify skipped: {exc}"

        return AnalysisResult(
            messages=messages,
            gaps=gaps,
            participants=participants,
            adapter_report=adapter_report,
            metadata={
                "min_severity": cfg.min_severity,
                "use_embeddings": use_embeddings,
                "use_llm_verify": use_llm_verify,
            },
        )

    # ------------------------------------------------------------------
    # Dedupe + validation
    # ------------------------------------------------------------------
    def _merge_and_validate(self, candidates: list[Gap], by_id: dict[int, Message]) -> list[Gap]:
        """Reject unknown ids; merge overlapping same-type candidates."""
        cfg = self.config
        valid: list[Gap] = []
        for g in candidates:
            if g.gap_type == "error":
                continue
            ids = [i for i in g.message_ids if i in by_id]
            if g.message_ids and not ids:
                continue  # reject gap whose ids are all invalid
            evidence = g.evidence or [self._quote(by_id[i], by_id) for i in ids]
            valid.append(
                Gap(
                    gap_type=g.gap_type,
                    severity=g.severity,
                    message_ids=ids,
                    evidence=evidence,
                    explanation=g.explanation,
                    score_parts=dict(g.score_parts),
                )
            )

        # Sort for deterministic merge: by type, then by first id
        valid.sort(key=lambda g: (g.gap_type, g.message_ids[0] if g.message_ids else 0))
        merged: list[Gap] = []
        for g in valid:
            placed = False
            for existing in merged:
                if existing.gap_type != g.gap_type:
                    continue
                if not g.message_ids or not existing.message_ids:
                    continue
                inter = len(set(g.message_ids) & set(existing.message_ids))
                union = len(set(g.message_ids) | set(existing.message_ids))
                if union and inter / union >= cfg.dedupe_overlap_frac:
                    # Merge: keep higher-severity framing, union ids/evidence
                    keep, drop = (existing, g) if existing.severity >= g.severity else (g, existing)
                    keep.message_ids = sorted(set(keep.message_ids) | set(drop.message_ids))
                    for ev in drop.evidence:
                        if ev not in keep.evidence:
                            keep.evidence.append(ev)
                    keep.score_parts = {**drop.score_parts, **keep.score_parts}
                    placed = True
                    break
            if not placed:
                merged.append(g)
        return merged

    # ------------------------------------------------------------------
    # Severity scoring
    # ------------------------------------------------------------------
    def _score(self, gaps: list[Gap], by_id: dict[int, Message], messages: list[Message]) -> list[Gap]:
        """Refine severity: time unanswered, re-asks, people involved, cue words."""
        cfg = self.config
        index = {m.id: i for i, m in enumerate(messages)}
        cue_words = tuple(cfg.deadline_cues) + tuple(cfg.closure_cues)

        for g in gaps:
            parts = dict(g.score_parts)
            ids = g.message_ids
            if not ids:
                g.severity = min(1.0, g.severity)
                continue

            first = by_id[ids[0]]
            last = by_id[ids[-1]]
            # Time gap between first flagged message and next non-flagged or last
            span_min = 0.0
            i0 = index.get(ids[0], 0)
            i1 = index.get(ids[-1], i0)
            if i1 > i0:
                span_min = (messages[i1].timestamp - messages[i0].timestamp).total_seconds() / 60.0

            time_factor = min(1.0, span_min / cfg.severity_time_cap_minutes)
            reasks = max(0, len(ids) - 1)
            reask_factor = min(1.0, reasks / cfg.severity_max_reasks)
            people = len({by_id[i].sender for i in ids if i in by_id})
            people_factor = min(1.0, (people - 1) / 3.0)
            text_blob = " ".join(by_id[i].text_lower for i in ids if i in by_id)
            cue_hit = 1.0 if any(c in text_blob for c in cue_words) else 0.0

            # Factors can *raise* severity above the detector prior, never crush it.
            factor_score = 0.25 * time_factor + 0.25 * reask_factor + 0.20 * people_factor + 0.30 * cue_hit
            blend = 0.55 * parts.get("base", g.severity) + 0.45 * factor_score
            final_sev = max(float(parts.get("base", g.severity)), min(1.0, blend))
            # Also allow pure-factor boosts for gaps whose prior is conservative
            final_sev = min(1.0, max(final_sev, 0.30 * factor_score + 0.25))
            g.score_parts = {
                **parts,
                "time_factor": round(time_factor, 3),
                "reask_factor": round(reask_factor, 3),
                "people_factor": round(people_factor, 3),
                "cue_hit": cue_hit,
                "span_minutes": round(span_min, 1),
                "people": float(people),
            }
            g.severity = round(max(g.severity * 0.5, final_sev), 3)
            # Refresh evidence to guarantee exact quotes with sender+timestamp
            g.evidence = [self._quote(by_id[i], by_id) for i in ids if i in by_id] or g.evidence
        return gaps

    @staticmethod
    def _quote(msg: Message, by_id: dict[int, Message]) -> str:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M") if msg.timestamp else "?"
        return f"[{msg.id}] {msg.sender} @ {ts}: {msg.text}"

    def _optional_embedded_clarification(self, messages: list[Message], candidates: list[Gap]) -> list[Gap]:
        """If sentence-transformers is installed and flag on, add re-ask gaps via embeddings."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np
        except Exception:
            return candidates  # package missing — keep TF-IDF candidates only

        cfg = self.config
        try:
            model = SentenceTransformer(cfg.embedding_model_name)
        except Exception:
            return candidates

        q_idxs = [i for i, m in enumerate(messages) if "?" in m.text]
        if len(q_idxs) < 2:
            return candidates
        texts = [messages[i].text for i in q_idxs]
        emb = model.encode(texts, normalize_embeddings=True)
        existing_pairs = {
            tuple(sorted(g.message_ids[:2]))
            for g in candidates
            if g.gap_type == "clarification" and len(g.message_ids) >= 2
        }
        new_gaps: list[Gap] = []
        for a in range(len(q_idxs)):
            for b in range(a + 1, len(q_idxs)):
                if messages[q_idxs[a]].sender != messages[q_idxs[b]].sender:
                    continue
                sim = float(np.dot(emb[a], emb[b]))
                pair = tuple(sorted((messages[q_idxs[a]].id, messages[q_idxs[b]].id)))
                if sim >= cfg.clarification_similarity_threshold and pair not in existing_pairs:
                    new_gaps.append(
                        Gap(
                            gap_type="clarification",
                            severity=round(min(0.9, 0.45 + 0.45 * sim), 2),
                            message_ids=list(pair),
                            evidence=[self._quote(messages[q_idxs[a]], {}), self._quote(messages[q_idxs[b]], {})],
                            explanation=(
                                f"Embedding model re-ask: {messages[q_idxs[b]].sender} asked a similar question "
                                f"(cosine {sim:.2f}) to message {messages[q_idxs[a]].id}."
                            ),
                            score_parts={"base": 0.45, "embedding_sim": round(0.45 * sim, 3)},
                        )
                    )
        return candidates + new_gaps


def analyze_conversation(messages: list[Message], **kwargs: Any) -> AnalysisResult:
    """Convenience wrapper used by app.py, evaluate.py, and CLI."""
    engine = EvidenceEngine(kwargs.pop("config", None))
    return engine.analyze(messages, **kwargs)
