"""PostgreSQL persistence: runs, answer_results, claim_results.

Constraints enforced:
- FK answer_results.run_id -> runs.id, claim_results.(run_id,answer_id) ->
  answer_results.(run_id,answer_id).
- UNIQUE (run_id, answer_id) and UNIQUE (run_id, answer_id, start_char, end_char).
- JSONB for config, metrics, and diagnostic id lists.
SQLAlchemy ORM => parameterized SQL. Schema created idempotently via
`create_all` (reproducible local setup; no manual DDL needed).
"""
from __future__ import annotations

import uuid
import re
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime, Float, ForeignKey, ForeignKeyConstraint, Integer, String, Text,
    UniqueConstraint, create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker,
)
from sqlalchemy.engine import make_url

from .config import get_settings


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    split: Mapped[str] = mapped_column(String(32))
    agent_version: Mapped[str] = mapped_column(String(16))
    prompt_version: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(64))
    evaluator_version: Mapped[str] = mapped_column(String(64))
    fact_version: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))  # completed | failed
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_hashes: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    answers: Mapped[list["AnswerRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class AnswerRow(Base):
    __tablename__ = "answer_results"
    __table_args__ = (
        UniqueConstraint("run_id", "answer_id", name="uq_run_answer"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE")
    )
    answer_id: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))  # success | empty | error
    error_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    turns_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="answers")
    claims: Mapped[list["ClaimRow"]] = relationship(
        back_populates="answer", cascade="all, delete-orphan"
    )


class ClaimRow(Base):
    __tablename__ = "claim_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "answer_id", "start_char", "end_char", name="uq_run_span"
        ),
        ForeignKeyConstraint(
            ["run_id", "answer_id"],
            ["answer_results.run_id", "answer_results.answer_id"],
            ondelete="CASCADE",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    answer_id: Mapped[str] = mapped_column(String(16))
    claim_id: Mapped[str] = mapped_column(String(64))
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(16))
    fact_ids: Mapped[list] = mapped_column(JSONB, default=list)
    reason: Mapped[str] = mapped_column(Text, default="")
    # Optional development match info (null on check runs / unmatched).
    match_info: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    answer: Mapped["AnswerRow"] = relationship(back_populates="claims")


_engine = None
_Session = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().database_url, future=True)
    return _engine


def get_session_factory():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine(), future=True, expire_on_commit=False)
    return _Session


def set_database_url(url):
    """Rebind the global engine/session to `url` (used by tests to target a
    disposable database). Disposes any previous engine."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(url, future=True)
    _Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
    return _engine


def current_url() -> str:
    return str(get_engine().url)


def init_db() -> None:
    """Create all tables idempotently."""
    Base.metadata.create_all(get_engine())


def drop_db(*, expected_database: str | None = None) -> None:
    """Drop all tables on the CURRENTLY BOUND engine.

    Require an explicitly named UUID test database, reject the configured app
    database, and verify the server target before issuing DDL.
    """
    engine = get_engine()
    app_db = make_url(get_settings().database_url).database
    active_db = engine.url.database
    if (not expected_database or not re.fullmatch(r"claimlens_test_[0-9a-f]{32}", expected_database)
            or active_db != expected_database or active_db == app_db):
        raise RuntimeError("refusing table deletion outside the explicitly named disposable database")
    # Verify the actual server target, not only a URL string (libpq options can
    # otherwise redirect a connection). Never use assert for destructive guards.
    with engine.begin() as connection:
        if connection.exec_driver_sql("SELECT current_database()").scalar_one() != expected_database:
            raise RuntimeError("connected database differs from disposable target")
        Base.metadata.drop_all(connection)
