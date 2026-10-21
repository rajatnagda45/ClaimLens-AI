"""Shared test fixtures and a deterministic FakeProvider.

Fakes are for controlled tests ONLY and never substitute for the four live
runs. The FakeProvider records every message it receives so tests can assert
that gold data never enters the model/tool payload.
"""
from __future__ import annotations

import json
import uuid

import psycopg
import pytest
from sqlalchemy.engine import make_url
from psycopg import sql

from claimlens import database
from claimlens.config import get_settings
from claimlens.provider import AssistantTurn, ProviderError, ProviderTimeout, ToolCall


def _connection_options(url, database):
    """Structured kwargs preserve passwords without string interpolation."""
    return {k: v for k, v in dict(dbname=database, host=url.host or "localhost",
                                port=url.port or 5432, user=url.username,
                                password=url.password, connect_timeout=5).items() if v is not None}


def pytest_addoption(parser):
    parser.addoption("--run-db", action="store_true", help="Create unique disposable local PostgreSQL databases for integration tests")


@pytest.fixture
def test_db(request):
    """Bind the ORM to a dedicated, disposable test database.

    Each fixture creates a fresh UUID database, verifies the server target, and
    removes only that database. Existing databases are never reused.
    """
    if not request.config.getoption("--run-db"):
        pytest.skip("PostgreSQL integration requires --run-db")
    app_url = make_url(get_settings().database_url)
    if app_url.host not in (None, "localhost", "127.0.0.1", "::1") or app_url.query:
        pytest.fail("Integration fixtures require a local URL without query overrides")
    app_db = app_url.database
    test_dbname = f"claimlens_test_{uuid.uuid4().hex}"

    # Safety: never operate on the application database.
    if test_dbname == app_db:
        raise RuntimeError("test database must differ from application database")
    # Never reuse an existing database. CREATE fails on collision.
    with psycopg.connect(**_connection_options(app_url, "postgres"), autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_dbname)))
    test_url = app_url.set(database=test_dbname)
    previous_engine, previous_session = database._engine, database._Session
    engine = None
    try:
        engine = database.set_database_url(test_url)
        with engine.connect() as conn:
            if conn.exec_driver_sql("SELECT current_database()").scalar_one() != test_dbname:
                raise RuntimeError("disposable target verification failed")
        database.init_db()
        yield test_url
    finally:
        if engine is not None:
            engine.dispose()
        database._engine, database._Session = previous_engine, previous_session
        # This exact UUID database was created above, never an existing target.
        with psycopg.connect(**_connection_options(app_url, "postgres"), autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(test_dbname)))


class FakeProvider:
    """Returns scripted AssistantTurns; optionally raises to simulate faults."""

    def __init__(self, turns=None, raises=None):
        self._turns = list(turns or [])
        self._raises = raises
        self.seen_messages: list[list[dict]] = []
        self.calls = 0

    def chat(self, messages, tools, tool_choice="required"):
        self.calls += 1
        self.last_tool_choice = tool_choice
        # Deep-ish copy of what the provider would actually see.
        self.seen_messages.append(json.loads(json.dumps(messages)))
        if self._raises is not None:
            raise self._raises
        if not self._turns:
            # Default: end without submit -> triggers invalid-output path.
            return AssistantTurn(content="done", tool_calls=[])
        return self._turns.pop(0)


def tool_call(name, args: dict, cid="call_1"):
    return ToolCall(id=cid, name=name, arguments=json.dumps(args))


def submit_turn(claims, cid="call_submit"):
    return AssistantTurn(tool_calls=[tool_call("submit_claims", {"claims": claims}, cid)])


def fact_then_submit(brand_id, claims):
    """Two turns: look up a brand, then submit claims."""
    return [
        AssistantTurn(tool_calls=[tool_call("get_brand_facts", {"brand_id": brand_id})]),
        submit_turn(claims),
    ]


@pytest.fixture
def fake_provider_cls():
    return FakeProvider
