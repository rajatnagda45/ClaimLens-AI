"""Checkpoint ordering, failed evidence, and concurrent report isolation."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from claimlens import runner as module
from claimlens.data_loader import Answer, GoldClaim
from claimlens.runner import Runner, RunFailure
from claimlens.schemas import AnswerResult, AnswerStatus
from .conftest import FakeProvider


@pytest.fixture
def isolated_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    answers = [Answer("X1", "check", "cleartrail", "q", "A."),
               Answer("X2", "check", "cleartrail", "q", "B.")]
    monkeypatch.setattr(module, "answers_for_split", lambda *a, **k: answers)
    runner = Runner(FakeProvider())
    monkeypatch.setattr(runner.agent, "run", lambda aid, *a: AnswerResult(
        answer_id=aid, status=AnswerStatus.EMPTY, latency_ms=1))
    return runner, tmp_path


def test_predictions_saved_before_gold_join(isolated_runner, monkeypatch):
    runner, folder = isolated_runner
    def gold():
        paths = list((folder / "runs").glob("*.json"))
        assert len(paths) == 1
        saved = json.loads(paths[0].read_text())
        assert len(saved["answers"]) == 2
        assert saved["metrics"]["joint_f1"] is None
        return [GoldClaim("X1", "X1:0:2", 0, 2, "A.", "unknown", [], "", [])]
    monkeypatch.setattr(module, "load_gold", lambda **k: gold())
    result = runner.run_split("v1", "development", persist=False)
    assert result["metadata"]["status"] == "completed"
    assert result["metrics"]["joint_f1"] == 0


def test_failure_retains_partial_predictions(isolated_runner, monkeypatch):
    runner, folder = isolated_runner
    def run(aid, *a):
        if aid == "X2":
            raise RuntimeError("DO_NOT_LEAK")
        return AnswerResult(answer_id=aid, status=AnswerStatus.EMPTY)
    monkeypatch.setattr(runner.agent, "run", run)
    with pytest.raises(RunFailure):
        runner.run_split("v1", "check", persist=False)
    saved = json.loads(next((folder / "runs").glob("*.json")).read_text())
    assert saved["metadata"]["status"] == "failed"
    assert len(saved["answers"]) == 1 and saved["metrics"]["joint_f1"] is None
    assert "DO_NOT_LEAK" not in json.dumps(saved)


def test_database_failure_keeps_disk_evidence(isolated_runner, monkeypatch):
    runner, folder = isolated_runner
    def fail(*a):
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(runner, "_persist", fail)
    with pytest.raises(RunFailure):
        runner.run_split("v1", "check")
    assert json.loads(next((folder / "runs").glob("*.json")).read_text())["metadata"]["status"] == "failed"


def test_concurrent_runs_do_not_overwrite_each_other(isolated_runner):
    runner, folder = isolated_runner
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda _: runner.run_split("v1", "check", persist=False), range(2)))
    assert len({r["metadata"]["run_id"] for r in reports}) == 2
    assert len(list((folder / "runs").glob("*.json"))) == 2


def test_check_never_loads_gold(isolated_runner, monkeypatch):
    runner, _ = isolated_runner
    def forbidden(**k):
        raise AssertionError("check must never load gold")
    monkeypatch.setattr(module, "load_gold", forbidden)
    result = runner.run_split("v2", "check", persist=False)
    assert result["metrics"]["gold_claims"] is None
    assert result["metrics"]["accepted_predictions"] == 0


def test_requires_evidence_destination(isolated_runner):
    runner, _ = isolated_runner
    with pytest.raises(ValueError, match="evidence destination"):
        runner.run_split("v1", "check", persist=False, write_report=False)
