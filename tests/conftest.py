"""Shared test fixtures and a deterministic FakeProvider.

Fakes are for controlled tests ONLY and never substitute for the four live
runs. The FakeProvider records every message it receives so tests can assert
that gold data never enters the model/tool payload.
"""
from __future__ import annotations

import json

import psycopg
import pytest
from sqlalchemy.engine import make_url

from claimlens import database
from claimlens.config import get_settings
from claimlens.provider import AssistantTurn, ProviderError, ProviderTimeout, ToolCall


def _admin_conninfo(url) -> str:
    """psycopg conninfo for the 'postgres' maintenance database."""
    parts = ["dbname=postgres", f"host={url.host or 'localhost'}",
             f"port={url.port or 5432}"]
    if url.username:
        parts.append(f"user={url.username}")
    if url.password:
        parts.append(f"password={url.password}")
    return " ".join(parts)


def _ensure_database(url, dbname: str) -> None:
    with psycopg.connect(_admin_conninfo(url), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')


@pytest.fixture
def test_db():
    """Bind the ORM to a dedicated, disposable test database.

    Explicit safety checks guarantee we never touch the configured application
    database: the test DB name must differ from the app DB name and must end in
    '_test'. drop_db() additionally refuses to run against the app DB.
    """
    app_url = make_url(get_settings().database_url)
    app_db = app_url.database
    test_dbname = f"{app_db}_test"

    # Safety: never operate on the application database.
    assert test_dbname != app_db
    assert test_dbname.endswith("_test")

    _ensure_database(app_url, test_dbname)
    test_url = app_url.set(database=test_dbname)
    database.set_database_url(str(test_url))

    # Safety: confirm the active binding is the disposable DB before any DDL.
    assert database.get_engine().url.database == test_dbname
    assert database.get_engine().url.database != app_db

    database.init_db()
    try:
        yield str(test_url)
    finally:
        database.drop_db()  # guarded; active DB is the test DB
        database.set_database_url(str(app_url))


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
