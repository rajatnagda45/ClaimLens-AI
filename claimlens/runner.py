"""Repeatable evaluation runs, persistence, and report generation.

A sequential runner processes a fixed CSV split, produces one terminal
AnswerResult per scheduled answer, persists runs/answer_results/claim_results,
scores the development split (null metrics for check), and writes JSON + a
concise Markdown report. Live and fake results are clearly distinguished by the
`provider` recorded in metadata.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION
from .agent import Agent
from .config import REPORTS_DIR, Settings, get_settings
from .data_loader import answers_for_split, load_gold, source_hashes
from .database import AnswerRow, ClaimRow, Run, get_session_factory, init_db
from .evaluator import check_split_metrics, score
from .prompts import PROMPT_VERSIONS
from .provider import Provider
from .schemas import AnswerResult, AnswerStatus, Metrics


def _match_info_map(answer_results, gold):
    """Per-claim dev match info: matched gold id + label correctness."""
    gold_by_key = {(g.answer_id, g.start_char, g.end_char): g for g in gold}
    taken: set = set()
    info: dict[str, dict] = {}
    for ar in answer_results:
        for c in ar.claims:
            key = (ar.answer_id, c.start_char, c.end_char)
            g = gold_by_key.get(key)
            if g and key not in taken and c.quote == g.quote:
                taken.add(key)
                info[c.claim_id] = {
                    "matched_gold_id": g.claim_id,
                    "label_correct": c.label == g.expected_label,
                    "expected_label": g.expected_label,
                }
            else:
                info[c.claim_id] = {"matched_gold_id": None}
    return info


class Runner:
    def __init__(self, provider: Provider, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.agent = Agent(provider, settings=self.settings)
        self.provider_name = type(provider).__name__

    def run_split(
        self, agent_version: str, split: str, persist: bool = True,
        write_report: bool = True,
    ) -> dict:
        answers = answers_for_split(split, settings=self.settings)
        results: list[AnswerResult] = []
        for a in answers:
            results.append(self.agent.run(
                a.answer_id, a.brand_id, a.question, a.ai_answer, agent_version,
            ))

        is_dev = split == "development"
        gold = load_gold(settings=self.settings) if is_dev else []
        gold = [g for g in gold if g.answer_id in {a.answer_id for a in answers}]
        metrics = score(results, gold) if is_dev else check_split_metrics(results)
        match_info = _match_info_map(results, gold) if is_dev else {}

        run_id = uuid.uuid4()
        meta = self._metadata(run_id, agent_version, split, metrics)

        if persist:
            self._persist(run_id, meta, results, metrics, is_dev, match_info)

        report = self._build_report(meta, results, metrics, is_dev, gold)
        if write_report:
            self._write_report(agent_version, split, report)
        return report

    def _metadata(self, run_id, agent_version, split, metrics) -> dict:
        return {
            "run_id": str(run_id),
            "split": split,
            "agent_version": agent_version,
            "prompt_version": PROMPT_VERSIONS[agent_version],
            "dataset_version": DATASET_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "fact_version": FACT_VERSION,
            "provider": self.provider_name,
            "model": self.settings.model,
            "settings": self.settings.masked(),
            "source_hashes": source_hashes(self.settings),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_live": self.provider_name == "OpenAIProvider",
        }

    def _persist(self, run_id, meta, results, metrics, is_dev, match_info):
        init_db()
        Session = get_session_factory()
        with Session() as s:
            run = Run(
                id=run_id, split=meta["split"], agent_version=meta["agent_version"],
                prompt_version=meta["prompt_version"],
                dataset_version=meta["dataset_version"],
                evaluator_version=meta["evaluator_version"],
                fact_version=meta["fact_version"], provider=meta["provider"],
                model=meta["model"], status=meta["status"],
                settings=meta["settings"], source_hashes=meta["source_hashes"],
                metrics=metrics.model_dump() if is_dev else _null_quality(metrics),
            )
            s.add(run)
            for ar in results:
                s.add(AnswerRow(
                    run_id=run_id, answer_id=ar.answer_id, status=ar.status.value,
                    error_type=ar.error_type.value if ar.error_type else None,
                    error_message=ar.error_message, latency_ms=ar.latency_ms,
                    turns_used=ar.turns_used, trace={"exchange": ar.trace},
                ))
                for c in ar.claims:
                    s.add(ClaimRow(
                        run_id=run_id, answer_id=ar.answer_id, claim_id=c.claim_id,
                        start_char=c.start_char, end_char=c.end_char, quote=c.quote,
                        label=c.label, fact_ids=c.fact_ids, reason=c.reason,
                        match_info=match_info.get(c.claim_id) if is_dev else None,
                    ))
            s.commit()

    def _build_report(self, meta, results, metrics, is_dev, gold) -> dict:
        return {
            "metadata": meta,
            "metrics": metrics.model_dump() if is_dev else _null_quality(metrics),
            "answers": [self._answer_report(ar) for ar in results],
        }

    @staticmethod
    def _answer_report(ar: AnswerResult) -> dict:
        return {
            "answer_id": ar.answer_id,
            "status": ar.status.value,
            "error_type": ar.error_type.value if ar.error_type else None,
            "error_message": ar.error_message,
            "latency_ms": ar.latency_ms,
            "turns_used": ar.turns_used,
            "claims": [c.model_dump() for c in ar.claims],
            "trace": ar.trace,
        }

    def _write_report(self, agent_version, split, report):
        REPORTS_DIR.mkdir(exist_ok=True)
        stem = f"{agent_version}_{split}"
        (REPORTS_DIR / f"{stem}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (REPORTS_DIR / f"{stem}.md").write_text(
            _markdown_summary(report), encoding="utf-8"
        )


def _null_quality(metrics: Metrics) -> dict:
    """Check-split metrics: gold-dependent fields null, counts preserved."""
    d = metrics.model_dump()
    for k in ("gold_claims", "accepted_predictions", "span_matches",
              "correct_label_matches", "extraction_precision",
              "extraction_recall", "extraction_f1", "joint_precision",
              "joint_recall", "joint_f1", "gold_coverage", "missed_gold_ids",
              "unmatched_prediction_ids", "wrong_label_ids"):
        d[k] = None
    return d


def _markdown_summary(report: dict) -> str:
    m = report["metadata"]
    mx = report["metrics"]
    lines = [
        f"# Run report: {m['agent_version']} / {m['split']}",
        "",
        f"- run_id: `{m['run_id']}`",
        f"- provider: **{m['provider']}** (live={m['is_live']})  model: `{m['model']}`",
        f"- prompt: `{m['prompt_version']}`  evaluator: `{m['evaluator_version']}`",
        f"- fact: `{m['fact_version']}`  dataset: `{m['dataset_version']}`",
        f"- timestamp: {m['timestamp']}",
        f"- source_hashes: {json.dumps(m['source_hashes'])}",
        "",
        "## Metrics",
    ]
    if mx.get("joint_f1") is None:
        lines.append("_Quality metrics are null for this split (check split)._")
    else:
        lines += [
            f"- G={mx['gold_claims']} P={mx['accepted_predictions']} "
            f"M={mx['span_matches']} C={mx['correct_label_matches']}",
            f"- extraction: P={mx['extraction_precision']:.3f} "
            f"R={mx['extraction_recall']:.3f} **F1={mx['extraction_f1']:.3f}**",
            f"- joint: P={mx['joint_precision']:.3f} "
            f"R={mx['joint_recall']:.3f} **F1={mx['joint_f1']:.3f}** (primary)",
            f"- gold coverage C/G = {mx['gold_coverage']:.3f}",
        ]
    counts = (f"- answers: success={mx['answers_success']} "
              f"empty={mx['answers_empty']} error={mx['answers_error']}")
    lines += ["", counts, "", "## Per-answer status", ""]
    for a in report["answers"]:
        extra = f" error={a['error_type']}" if a["error_type"] else ""
        lines.append(
            f"- {a['answer_id']}: {a['status']} "
            f"claims={len(a['claims'])} latency={a['latency_ms']:.0f}ms{extra}"
        )
    return "\n".join(lines) + "\n"
