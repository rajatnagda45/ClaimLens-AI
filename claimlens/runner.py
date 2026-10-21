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
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION
from .agent import Agent
from .config import REPORTS_DIR, PROJECT_ROOT, Settings, get_settings
from .data_loader import answers_for_split, load_gold, source_hashes
from .database import AnswerRow, ClaimRow, Run, get_session_factory, init_db
from .evaluator import check_split_metrics, score
from .prompts import PROMPT_VERSIONS
from .provider import Provider
from .schemas import AnswerResult, AnswerStatus, Metrics


class RunFailure(RuntimeError):
    def __init__(self, run_id):
        self.run_id = str(run_id)
        super().__init__("Evaluation infrastructure failed; partial evidence retained where available")


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
        if agent_version not in PROMPT_VERSIONS:
            raise ValueError("unknown agent version")
        if not persist and not write_report:
            raise ValueError("evaluation requires a prediction evidence destination")
        results: list[AnswerResult] = []
        is_dev = split == "development"
        run_id = uuid.uuid4()
        metrics = Metrics()
        meta = self._metadata(run_id, agent_version, split, metrics)
        meta["status"] = "running"
        meta["scheduled_answer_ids"] = [a.answer_id for a in answers]
        match_info = {}

        def checkpoint():
            report = self._build_report(meta, results, metrics, is_dev, [])
            # Disk evidence survives DB failure. Unique run paths cannot
            # overwrite the frozen historical four-run reports.
            if write_report:
                self._write_report(agent_version, split, report)
            if persist:
                self._persist(run_id, meta, results, metrics, is_dev, match_info)
            return report

        try:
            checkpoint()
            for a in answers:
                results.append(self.agent.run(
                    a.answer_id, a.brand_id, a.question, a.ai_answer, agent_version,
                ))
                metrics = check_split_metrics(results)
                checkpoint()  # save predictions BEFORE any gold join
            if source_hashes(self.settings) != meta["source_hashes"]:
                raise ValueError("source inputs changed during the run")
            kwargs = {"scheduled_ids": meta["scheduled_answer_ids"],
                      "answer_texts": {a.answer_id: a.ai_answer for a in answers}}
            if is_dev:
                gold = load_gold(settings=self.settings)
                metrics = score(results, gold, **kwargs)
                match_info = _match_info_map(results, gold)
            else:
                metrics = check_split_metrics(results, **kwargs)
            meta["status"] = "completed"
            meta["completed_at"] = datetime.now(timezone.utc).isoformat()
            return checkpoint()
        except BaseException as exc:
            meta["status"] = "failed"
            meta["failure"] = "Evaluation interrupted or infrastructure failed"
            metrics = Metrics(accepted_predictions=sum(len(a.claims) for a in results),
                              answers_success=sum(a.status == AnswerStatus.SUCCESS for a in results),
                              answers_empty=sum(a.status == AnswerStatus.EMPTY for a in results),
                              answers_error=sum(a.status == AnswerStatus.ERROR for a in results))
            match_info = {}
            try:
                checkpoint()
            except Exception:
                pass  # a failed storage destination cannot promise durability
            if not isinstance(exc, Exception):
                raise
            raise RunFailure(run_id) from exc

    def _metadata(self, run_id, agent_version, split, metrics) -> dict:
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
                text=True, timeout=2,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        runtime = hashlib.sha256()
        for path in sorted((PROJECT_ROOT / "claimlens").rglob("*.py")):
            runtime.update(str(path.relative_to(PROJECT_ROOT)).encode())
            runtime.update(path.read_bytes())
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
            "source_revision": revision,
            "runtime_sha256": runtime.hexdigest(),
            "prompt_sha256": hashlib.sha256((PROJECT_ROOT / "claimlens" / "prompts" / f"{agent_version}.py").read_bytes()).hexdigest(),
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_live": self.provider_name == "OpenAIProvider",
        }

    def _persist(self, run_id, meta, results, metrics, is_dev, match_info):
        init_db()
        Session = get_session_factory()
        with Session() as s:
            run = s.get(Run, run_id)
            if run is None:
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
                s.flush()
            run.status = meta["status"]
            run.metrics = metrics.model_dump() if is_dev else _null_quality(metrics)
            run.source_revision = meta.get("source_revision")
            run.settings = {
                **meta["settings"],
                "run_provenance": {k: meta.get(k) for k in (
                    "runtime_sha256", "prompt_sha256", "scheduled_answer_ids", "failure"
                )},
            }
            for ar in results:
                existing = s.query(AnswerRow).filter_by(
                    run_id=run_id, answer_id=ar.answer_id
                ).first()
                if existing is None:
                    s.add(AnswerRow(
                        run_id=run_id, answer_id=ar.answer_id, status=ar.status.value,
                        error_type=ar.error_type.value if ar.error_type else None,
                        error_message=ar.error_message, latency_ms=ar.latency_ms,
                        turns_used=ar.turns_used, trace={"exchange": ar.trace},
                    ))
                    s.flush()
                for c in ar.claims:
                    saved = s.query(ClaimRow).filter_by(
                        run_id=run_id, answer_id=ar.answer_id,
                        start_char=c.start_char, end_char=c.end_char,
                    ).first()
                    if saved is None:
                        saved = ClaimRow(
                            run_id=run_id, answer_id=ar.answer_id, claim_id=c.claim_id,
                            start_char=c.start_char, end_char=c.end_char, quote=c.quote,
                            label=c.label, fact_ids=c.fact_ids, reason=c.reason,
                        )
                        s.add(saved)
                    saved.match_info = match_info.get(c.claim_id) if is_dev else None
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
        folder = REPORTS_DIR / "runs"
        folder.mkdir(parents=True, exist_ok=True)
        stem = str(uuid.UUID(report["metadata"]["run_id"]))
        for suffix, content in (("json", json.dumps(report, indent=2)),
                                ("md", _markdown_summary(report))):
            target = folder / f"{stem}.{suffix}"
            temp = target.with_suffix(f".{suffix}.tmp")
            temp.write_text(content, encoding="utf-8")
            temp.replace(target)


def _null_quality(metrics: Metrics) -> dict:
    """Check-split metrics: gold-dependent fields null, counts preserved."""
    d = metrics.model_dump()
    for k in ("gold_claims", "span_matches",
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
        f"- status: {m['status']}",
        f"- source_hashes: {json.dumps(m['source_hashes'])}",
        "",
        "## Metrics",
    ]
    if mx.get("joint_f1") is None:
        lines.append("_Quality metrics are unavailable for check, empty, running, or failed runs._")
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
            f"claims={len(a['claims'])} latency={a['latency_ms'] or 0:.0f}ms{extra}"
        )
    return "\n".join(lines) + "\n"
