"""Failure categorization and v1-vs-v2 comparison from saved reports.

Operates only on SAVED report JSON + gold (development). Never influences the
inference path. Produces an evidence-based breakdown of where predictions
diverge from gold so the targeted v2 change is justified by real data.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import REPORTS_DIR
from .data_loader import load_gold, load_answers, source_hashes
from .evaluator import score, check_split_metrics
from .schemas import AnswerResult


def verify_report(report_path: str | Path) -> dict:
    """Recompute saved results without inference, writes, or check-set labels."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    meta = report["metadata"]
    if meta["split"] not in {"development", "check"} or meta["status"] != "completed":
        raise ValueError("only complete, known splits may be evaluated")
    if meta["source_hashes"] != source_hashes():
        raise ValueError("report source hashes differ from current inputs")
    answers = [a for a in load_answers() if a.split == meta["split"]]
    results = [AnswerResult.model_validate(a) for a in report["answers"]]
    kwargs = {"scheduled_ids": [a.answer_id for a in answers],
              "answer_texts": {a.answer_id: a.ai_answer for a in answers}}
    if meta["split"] == "development":
        metrics = score(results, load_gold(), **kwargs).model_dump()
    else:
        metrics = check_split_metrics(results, **kwargs).model_dump()
    for key, value in metrics.items():
        # Historical check reports nulled P as well as quality. Accept that
        # legacy count omission, but never a non-null check quality score.
        if key == "accepted_predictions" and meta["split"] == "check" and report["metrics"].get(key) is None:
            continue
        if report["metrics"].get(key) != value:
            raise ValueError(f"saved metric mismatch: {key}")
    return {"report": report, "metrics": metrics}


def _gold_index():
    g = load_gold()
    by_key = {(x.answer_id, x.start_char, x.end_char): x for x in g}
    by_answer: dict[str, list] = {}
    for x in g:
        by_answer.setdefault(x.answer_id, []).append(x)
    return g, by_key, by_answer


def categorize(report_path: str | Path) -> dict:
    """Categorize divergences between a dev report's predictions and gold."""
    verified = verify_report(report_path)
    report = verified["report"]
    if report["metadata"]["split"] != "development":
        raise ValueError("failure analysis requires development data")
    gold, gold_by_key, gold_by_answer = _gold_index()

    cats = {
        "missed_factual_sentence": [],   # gold span with no matching prediction
        "unmatched_prediction": [],     # may be advice OR a boundary error
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
            if g is None or g.quote != c["quote"]:
                cats["unmatched_prediction"].append({"claim_id": c["claim_id"],
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
            "metrics": verified["metrics"]}


def compare(v1_report: str | Path, v2_report: str | Path) -> dict:
    reports = [verify_report(p)["report"] for p in (v1_report, v2_report)]
    for field in ("split", "dataset_version", "evaluator_version", "fact_version",
                  "provider", "model", "settings", "source_hashes"):
        if reports[0]["metadata"][field] != reports[1]["metadata"][field]:
            raise ValueError(f"uncontrolled comparison: {field}")
    if [r["metadata"]["agent_version"] for r in reports] != ["v1", "v2"]:
        raise ValueError("comparison requires v1 followed by v2")
    a1 = categorize(v1_report)
    a2 = categorize(v2_report)
    m1, m2 = a1["metrics"], a2["metrics"]
    gold = {(g.answer_id, g.start_char, g.end_char): g for g in load_gold()}
    def correct_ids(report):
        return {c["claim_id"] for a in report["answers"] for c in a["claims"]
                if (g := gold.get((a["answer_id"], c["start_char"], c["end_char"])))
                and g.quote == c["quote"] and g.expected_label == c["label"]}
    c1, c2 = map(correct_ids, reports)
    return {
        "v1": {"joint_f1": m1["joint_f1"], "extraction_f1": m1["extraction_f1"],
               "summary": a1["summary"], "confusion": a1["confusion"]},
        "v2": {"joint_f1": m2["joint_f1"], "extraction_f1": m2["extraction_f1"],
               "summary": a2["summary"], "confusion": a2["confusion"]},
        "delta_joint_f1": (m2["joint_f1"] - m1["joint_f1"]),
        "delta_extraction_f1": (m2["extraction_f1"] - m1["extraction_f1"]),
        "newly_correct_ids": sorted(c2 - c1),
        "lost_correct_ids": sorted(c1 - c2),
    }


if __name__ == "__main__":
    out = compare(REPORTS_DIR / "v1_development.json",
                  REPORTS_DIR / "v2_development.json")
    print(json.dumps(out, indent=2))
