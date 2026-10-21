"""Validated configuration and secret handling.

All provider credentials and tunables come from environment variables so that
no secret is ever hardcoded, printed, or committed. See `.env.example`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()  # load .env if present; real secrets never committed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
REPORTS_DIR = PROJECT_ROOT / "reports"


class Settings(BaseModel):
    """Runtime settings sourced from the environment."""

    # Provider / model
    openai_api_key: str = Field(default="", repr=False)
    model: str = Field(default="gpt-4o-mini")
    provider: str = Field(default="openai")

    # Execution constraints (assessment: <=3 turns, configurable timeout)
    max_turns: int = Field(default=3, ge=1, le=3)
    request_timeout_s: float = Field(default=90.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://localhost:5432/claimlens"
    )

    # Data files
    answers_csv: str = Field(default=str(DATA_DIR / "answers.csv"))
    brand_facts_csv: str = Field(default=str(DATA_DIR / "brand_facts.csv"))
    development_gold_csv: str = Field(default=str(DATA_DIR / "development_gold.csv"))

    def masked(self) -> dict:
        """Return a secret-free view suitable for logs and run metadata."""
        d = self.model_dump()
        d["openai_api_key"] = "***set***" if self.openai_api_key else "***unset***"
        # Never serialize connection credentials or credential-bearing query
        # parameters. The database target is deployment config, not inference.
        d["database_url"] = "[redacted]"
        return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("CLAIMLENS_MODEL", "gpt-4o-mini"),
        provider=os.getenv("CLAIMLENS_PROVIDER", "openai"),
        max_turns=int(os.getenv("CLAIMLENS_MAX_TURNS", "3")),
        request_timeout_s=float(os.getenv("CLAIMLENS_TIMEOUT_S", "90")),
        temperature=float(os.getenv("CLAIMLENS_TEMPERATURE", "0")),
        database_url=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://localhost:5432/claimlens"
        ),
    )
