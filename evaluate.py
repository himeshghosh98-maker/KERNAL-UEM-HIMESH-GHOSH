"""Evaluate GapDetect on labeled sample chats.

Runs the evidence engine on every data/*.txt chat, compares predictions to
data/*.labels.json, and writes precision/recall/F1 per gap type to results.md.

A prediction is correct if its message ids overlap a labeled gap of the same type.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import DEFAULT_CONFIG, DetectorConfig
from core.engine import EvidenceEngine
from core.parser import parse_file


def load_labels(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("gaps", [])


def evaluate_chat(
    chat_path: Path,
    labels_path: Path,
    config: DetectorConfig | None = None,
) -> dict:
    messages = parse_file(chat_path)
    labels = load_labels(labels_path)
    result = EvidenceEngine(config or DEFAULT_CONFIG).analyze(messages, use_adapters=True)

    by_type_labels: dict[str, list[set[int]]] = {}
    for lab in labels:
        by_type_labels.setdefault(lab["gap_type"], []).append(set(lab["message_ids"]))

    by_type_preds: dict[str, list[set[int]]] = {}
    for g in result.gaps:
        by_type_preds.setdefault(g.gap_type, []).append(set(g.message_ids))

    rows = []
    for gap_type in sorted(set(by_type_labels) | set(by_type_preds)):
        gold = by_type_labels.get(gap_type, [])
        preds = by_type_preds.get(gap_type, [])
        matched_gold: set[int] = set()
        matched_pred: set[int] = set()
        for pi, pred in enumerate(preds):
            for gi, lab in enumerate(gold):
                if pred & lab:
                    matched_pred.add(pi)
                    matched_gold.add(gi)
                    break
        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(gold) - len(matched_gold)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "gap_type": gap_type,
                "gold": len(gold),
                "pred": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"chat": chat_path.stem, "rows": rows, "n_messages": len(messages)}


def micro_f1(all_rows: list[dict]) -> tuple[float, float, float]:
    tp = sum(r["tp"] for r in all_rows)
    fp = sum(r["fp"] for r in all_rows)
    fn = sum(r["fn"] for r in all_rows)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate GapDetect on labeled chats")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results.md"))
    ap.add_argument("--label-threshold", type=float, default=None, help="Override min_severity")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cfg = DetectorConfig()
    if args.label_threshold is not None:
        cfg.min_severity = args.label_threshold

    chats = sorted(data_dir.glob("*.txt"))
    if not chats:
        print(f"No .txt chats found in {data_dir}")
        sys.exit(1)

    chat_results = []
    for chat in chats:
        labels_path = chat.with_suffix(".labels.json")
        if not labels_path.exists():
            print(f"SKIP {chat.name}: no labels file")
            continue
        chat_results.append(evaluate_chat(chat, labels_path, cfg))

    lines = [
        "# GapDetect Evaluation Results",
        "",
        f"Config: min_severity={cfg.min_severity}, "
        f"unanswered_window={cfg.unanswered_lookahead_messages} msgs / "
        f"{cfg.unanswered_lookahead_minutes:.0f} min, "
        f"clar_sim={cfg.clarification_similarity_threshold}, "
        f"topic_shared_tokens={cfg.topic_min_shared_tokens}",
        "",
        "A prediction counts as a true positive if its message ids overlap a labeled gap of the same type.",
        "",
    ]

    all_rows: list[dict] = []
    for res in chat_results:
        lines.append(f"## {res['chat']} ({res['n_messages']} messages)")
        lines.append("")
        lines.append("| Gap Type | Gold | Pred | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("|----------|-----:|-----:|---:|---:|---:|----------:|-------:|---:|")
        for row in res["rows"]:
            all_rows.append(row)
            lines.append(
                f"| {row['gap_type']} | {row['gold']} | {row['pred']} | {row['tp']} | "
                f"{row['fp']} | {row['fn']} | {row['precision']:.2f} | "
                f"{row['recall']:.2f} | {row['f1']:.2f} |"
            )
        lines.append("")

    # Per gap-type micro aggregate
    lines.append("## Aggregate by gap type (all chats)")
    lines.append("")
    lines.append("| Gap Type | Gold | Pred | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|----------|-----:|-----:|---:|---:|---:|----------:|-------:|---:|")
    types = sorted({r["gap_type"] for r in all_rows})
    for t in types:
        rows_t = [r for r in all_rows if r["gap_type"] == t]
        p, r, f = micro_f1(rows_t)
        gold = sum(r_["gold"] for r_ in rows_t)
        pred = sum(r_["pred"] for r_ in rows_t)
        tp = sum(r_["tp"] for r_ in rows_t)
        fp = sum(r_["fp"] for r_ in rows_t)
        fn = sum(r_["fn"] for r_ in rows_t)
        lines.append(f"| {t} | {gold} | {pred} | {tp} | {fp} | {fn} | {p:.2f} | {r:.2f} | {f:.2f} |")
    p_all, r_all, f_all = micro_f1(all_rows)
    lines.append("")
    lines.append(f"**Micro overall:** precision={p_all:.2f} recall={r_all:.2f} F1={f_all:.2f}")
    lines.append("")
    lines.append("_Thresholds tuned once after this run — see README Results section for before/after._")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Micro overall: P={p_all:.2f} R={r_all:.2f} F1={f_all:.2f}")


if __name__ == "__main__":
    main()
