"""Database isolation and persistence tests on a disposable test database.

These confirm (a) the fixture binds a dedicated '<app>_test' database and never
the application database, (b) drop_db() refuses to run against the application
database, and (c) nonempty claims, valid empty outputs, and answer-level errors
all persist and read back correctly — using synthetic inputs, no API credits.
"""
import uuid

import pytest
from sqlalchemy.engine import make_url

from claimlens import database
from claimlens.config import get_settings
from claimlens.database import AnswerRow, ClaimRow, Run, get_session_factory
from claimlens.runner import Runner
from claimlens.schemas import (
    AnswerResult, AnswerStatus, Claim, ErrorType, Metrics,
)

from .conftest import FakeProvider


def test_fixture_binds_disposable_db_not_app_db(test_db):
    app_db = make_url(get_settings().database_url).database
    active = database.get_engine().url.database
    assert active == f"{app_db}_test"
    assert active != app_db


def test_drop_db_refuses_application_database():
    # Bind the global engine to the real application URL and confirm drop_db
    # raises instead of dropping it. No tables are dropped.
    database.set_database_url(get_settings().database_url)
    try:
        with pytest.raises(RuntimeError):
            database.drop_db()
    finally:
        database.set_database_url(get_settings().database_url)


def _claim(aid, s, e, q, label="unknown"):
    return Claim(claim_id=f"{aid}:{s}:{e}", start_char=s, end_char=e,
                 quote=q, label=label, fact_ids=[], reason="")


def test_persist_nonempty_empty_and_error_outcomes(test_db):
    runner = Runner(FakeProvider())
    results = [
        AnswerResult(answer_id="A01", status=AnswerStatus.SUCCESS,
                     claims=[_claim("A01", 0, 5, "abcd.", "supported")],
                     latency_ms=1.0),
        AnswerResult(answer_id="A02", status=AnswerStatus.EMPTY, claims=[],
                     latency_ms=1.0),
        AnswerResult(answer_id="A03", status=AnswerStatus.ERROR,
                     error_type=ErrorType.PROVIDER_TIMEOUT,
                     error_message="timed out", latency_ms=1.0),
    ]
    rid = uuid.uuid4()
    meta = runner._metadata(rid, "v1", "check", Metrics())
    runner._persist(rid, meta, results, Metrics(), is_dev=False, match_info={})

    with get_session_factory()() as s:
        assert s.get(Run, rid) is not None
        rows = s.query(AnswerRow).filter(AnswerRow.run_id == rid).all()
        status = {r.answer_id: r.status for r in rows}
        assert status == {"A01": "success", "A02": "empty", "A03": "error"}
        err = next(r for r in rows if r.answer_id == "A03")
        assert err.error_type == "provider_timeout"
        claims = s.query(ClaimRow).filter(ClaimRow.run_id == rid).all()
        assert len(claims) == 1
        assert claims[0].answer_id == "A01" and claims[0].label == "supported"


def test_unique_span_constraint_enforced(test_db):
    """Duplicate (run_id, answer_id, start_char, end_char) is rejected by PG."""
    import sqlalchemy
    database.init_db()
    rid = uuid.uuid4()
    with get_session_factory()() as s:
        s.add(Run(id=rid, split="check", agent_version="v1",
                  prompt_version="p", dataset_version="d", evaluator_version="e",
                  fact_version="f", provider="FakeProvider", model="m",
                  status="completed", settings={}, source_hashes={}, metrics={}))
        s.add(AnswerRow(run_id=rid, answer_id="A01", status="success"))
        s.commit()
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        with get_session_factory()() as s:
            for _ in range(2):
                s.add(ClaimRow(run_id=rid, answer_id="A01", claim_id="A01:0:5",
                               start_char=0, end_char=5, quote="abcd.",
                               label="unknown", fact_ids=[], reason=""))
            s.commit()
