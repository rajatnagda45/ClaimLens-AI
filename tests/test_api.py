"""API and persistence tests with a FakeProvider override."""
import pytest
from fastapi.testclient import TestClient

from claimlens import api
from claimlens.agent import Agent
from claimlens.database import drop_db, init_db
from claimlens.provider import AssistantTurn

from .conftest import FakeProvider, submit_turn, tool_call


@pytest.fixture(scope="module", autouse=True)
def _db():
    init_db()
    yield
    # leave tables for inspection; drop to keep tests hermetic
    drop_db()


@pytest.fixture
def client():
    return TestClient(api.app)


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr(api, "get_provider", lambda: provider)


def test_classify_success(client, monkeypatch):
    claims = [{"quote": "The Mini is a nice speaker.", "label": "unknown",
               "fact_ids": [], "reason": "n."}]
    _patch_provider(monkeypatch, FakeProvider([submit_turn(claims)]))
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


def test_get_run_missing_404(client):
    import uuid
    assert client.get(f"/evaluation-runs/{uuid.uuid4()}").status_code == 404


def test_persist_and_retrieve_run(monkeypatch):
    """Run a tiny development-like persistence cycle and join-read it back."""
    from claimlens.runner import Runner
    claims = [{"quote": "The Mini is a nice speaker.", "label": "unknown",
               "fact_ids": [], "reason": "n."}]
    # Script: every answer returns empty to keep it fast & deterministic.
    class AllEmpty(FakeProvider):
        def chat(self, messages, tools, tool_choice="required"):
            self.seen_messages.append(messages)
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
