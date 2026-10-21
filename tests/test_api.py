"""API and persistence tests with a FakeProvider override.

All tests run against a dedicated, disposable test database (the `test_db`
fixture). The application database is never touched; see conftest.py and the
drop_db() safety guard.
"""
import pytest
from fastapi.testclient import TestClient

from claimlens import api
from claimlens.provider import AssistantTurn, ProviderError, ProviderTimeout

from .conftest import FakeProvider, submit_turn, tool_call, fact_then_submit


@pytest.fixture
def client():
    return TestClient(api.app)


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr(api, "get_provider", lambda: provider)


def test_classify_success(client, monkeypatch):
    claims = [{"quote": "The Mini is a nice speaker.", "label": "unknown",
               "fact_ids": [], "reason": "n."}]
    _patch_provider(monkeypatch, FakeProvider(fact_then_submit("cleartrail", claims)))
    r = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "q",
        "ai_answer": "The Mini is a nice speaker.", "agent_version": "v1",
    })
    assert r.status_code == 200
    assert r.json()["claims"][0]["label"] == "unknown"


def test_classify_rejects_gold_fields_422(client):
    r = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "q",
        "ai_answer": "x.", "agent_version": "v1",
        "expected_label": "supported",  # forbidden extra field
    })
    assert r.status_code == 422


def test_classify_unknown_brand_404(client):
    r = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "sony", "question": "q",
        "ai_answer": "x.", "agent_version": "v1",
    })
    assert r.status_code == 404


def test_classify_provider_error_502(client, monkeypatch):
    turn = AssistantTurn(tool_calls=[tool_call("evil", {})])
    _patch_provider(monkeypatch, FakeProvider([turn]))
    r = client.post("/classify", json={
        "answer_id": "A01", "brand_id": "cleartrail", "question": "q",
        "ai_answer": "The Mini is nice.", "agent_version": "v1",
    })
    assert r.status_code == 502


def test_get_run_invalid_uuid_422(client):
    assert client.get("/evaluation-runs/not-a-uuid").status_code == 422


def test_get_run_missing_404(client, test_db):
    import uuid
    assert client.get(f"/evaluation-runs/{uuid.uuid4()}").status_code == 404


def test_persist_and_retrieve_run(monkeypatch, test_db):
    """Run a tiny development-like persistence cycle and join-read it back."""
    from claimlens.runner import Runner
    claims = [{"quote": "The Mini is a nice speaker.", "label": "unknown",
               "fact_ids": [], "reason": "n."}]
    # Script: every answer returns empty to keep it fast & deterministic.
    class AllEmpty(FakeProvider):
        def chat(self, messages, tools, tool_choice="required"):
            self.seen_messages.append(messages)
            if not any(m["role"] == "tool" for m in messages):
                return fact_then_submit("cleartrail", [])[0]
            return submit_turn([])
    runner = Runner(AllEmpty())
    report = runner.run_split("v1", "development", persist=True, write_report=False)
    rid = report["metadata"]["run_id"]

    client = TestClient(api.app)
    r = client.get(f"/evaluation-runs/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert body["split"] == "development"
    assert len(body["answers"]) == 40  # every scheduled answer persisted
    # metrics present for development
    assert body["metrics"]["gold_claims"] == 376


@pytest.mark.parametrize("error,status", [(ProviderError("secret"), 502), (ProviderTimeout("secret"), 504)])
def test_provider_failures_sanitized(client, monkeypatch, error, status):
    _patch_provider(monkeypatch, FakeProvider(raises=error))
    response = client.post("/classify", json={"answer_id": "X", "brand_id": "cleartrail",
                           "question": "q", "ai_answer": "A.", "agent_version": "v1"})
    assert response.status_code == status
    assert "secret" not in response.text


def test_provider_setup_failure(client, monkeypatch):
    def fail():
        raise ProviderError("DO_NOT_LEAK")
    monkeypatch.setattr(api, "OpenAIProvider", fail)
    r = client.post("/classify", json={"answer_id": "X", "brand_id": "cleartrail",
                     "question": "q", "ai_answer": "A.", "agent_version": "v1"})
    assert r.status_code == 502 and "DO_NOT_LEAK" not in r.text


def test_unknown_model_lookup_is_upstream_failure(client, monkeypatch):
    _patch_provider(monkeypatch, FakeProvider([AssistantTurn(tool_calls=[
        tool_call("get_brand_facts", {"brand_id": "unknown"})])]))
    r = client.post("/classify", json={"answer_id": "X", "brand_id": "cleartrail",
                     "question": "q", "ai_answer": "A.", "agent_version": "v1"})
    assert r.status_code == 502


def test_startup_against_disposable_database(test_db):
    with TestClient(api.app) as client:
        assert client.get("/docs").status_code == 200


def test_evaluation_failure_returns_reference(client, monkeypatch):
    from claimlens.runner import RunFailure
    import uuid
    rid = uuid.uuid4()
    _patch_provider(monkeypatch, FakeProvider())
    def fail(*a, **k):
        raise RunFailure(rid)
    monkeypatch.setattr(api.Runner, "run_split", fail)
    r = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert r.status_code == 503 and r.json()["run_id"] == str(rid)
