"""Failure categorization and v1-vs-v2 comparison from saved reports.

Operates only on SAVED report JSON + gold (development). Never influences the
inference path. Produces an evidence-based breakdown of where predictions
diverge from gold so the targeted v2 change is justified by real data.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import REPORTS_DIR
from .data_loader import load_gold


def _gold_index():
    g = load_gold()
    by_key = {(x.answer_id, x.start_char, x.end_char): x for x in g}
    by_answer: dict[str, list] = {}
    for x in g:
        by_answer.setdefault(x.answer_id, []).append(x)
    return g, by_key, by_answer


def categorize(report_path: str | Path) -> dict:
    """Categorize divergences between a dev report's predictions and gold."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    gold, gold_by_key, gold_by_answer = _gold_index()

    cats = {
        "missed_factual_sentence": [],   # gold span with no matching prediction
        "extra_non_factual": [],         # prediction span not in gold
        "wrong_label": [],               # span matches, label differs
        "answer_error": [],              # answer-level error (all gold missed)
    }
    matched_keys = set()
    for a in report["answers"]:
        aid = a["answer_id"]
        if a["status"] == "error":
            cats["answer_error"].append({"answer_id": aid, "error": a["error_type"]})
            continue
        for c in a["claims"]:
            key = (aid, c["start_char"], c["end_char"])
            g = gold_by_key.get(key)
            if g is None:
                cats["extra_non_factual"].append({"claim_id": c["claim_id"],
                                                  "quote": c["quote"]})
            else:
                matched_keys.add(key)
                if c["label"] != g.expected_label:
                    cats["wrong_label"].append({
                        "claim_id": c["claim_id"], "predicted": c["label"],
                        "expected": g.expected_label,
                        "tags": g.difficulty_tags,
                    })
    for g in gold:
        key = (g.answer_id, g.start_char, g.end_char)
        if key not in matched_keys:
            cats["missed_factual_sentence"].append({
                "claim_id": g.claim_id, "expected": g.expected_label,
                "tags": g.difficulty_tags,
            })

    summary = {k: len(v) for k, v in cats.items()}
    # Label-confusion matrix among matched-but-wrong predictions.
    confusion: dict[str, int] = {}
    for w in cats["wrong_label"]:
        key = f"{w['expected']}->{w['predicted']}"
        confusion[key] = confusion.get(key, 0) + 1
    return {"summary": summary, "confusion": confusion, "detail": cats,
            "metrics": report["metrics"]}


def compare(v1_report: str | Path, v2_report: str | Path) -> dict:
    a1 = categorize(v1_report)
    a2 = categorize(v2_report)
    m1, m2 = a1["metrics"], a2["metrics"]
    return {
        "v1": {"joint_f1": m1["joint_f1"], "extraction_f1": m1["extraction_f1"],
               "summary": a1["summary"], "confusion": a1["confusion"]},
        "v2": {"joint_f1": m2["joint_f1"], "extraction_f1": m2["extraction_f1"],
               "summary": a2["summary"], "confusion": a2["confusion"]},
        "delta_joint_f1": (m2["joint_f1"] - m1["joint_f1"]),
        "delta_extraction_f1": (m2["extraction_f1"] - m1["extraction_f1"]),
    }


if __name__ == "__main__":
    out = compare(REPORTS_DIR / "v1_development.json",
                  REPORTS_DIR / "v2_development.json")
    print(json.dumps(out, indent=2))
