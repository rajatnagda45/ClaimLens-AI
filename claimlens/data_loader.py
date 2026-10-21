"""CSV loading, schema validation, source hashing, split selection.

Rules enforced here (assessment "Data integrity"):
- Use Python's csv parser with UTF-8 decoding; quoted commas and embedded
  newlines are DATA, never record separators.
- `ai_answer` is never normalized/rewritten before offsets are computed.
- SHA-256 of each source file is recorded for run metadata.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings


@dataclass(frozen=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str  # exact decoded text — never altered


@dataclass(frozen=True)
class BrandFact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: str
    value_type: str  # number | boolean | string
    unit: str
    condition: str
    valid_from: str
    valid_to: str


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: list[str]
    explanation: str
    difficulty_tags: list[str]


ANSWER_COLS = ["answer_id", "split", "brand_id", "question", "ai_answer"]
FACT_COLS = [
    "fact_id", "brand_id", "model_id", "field", "value", "value_type",
    "unit", "condition", "valid_from", "valid_to",
]
GOLD_COLS = [
    "answer_id", "claim_id", "start_char", "end_char", "quote",
    "expected_label", "evidence_fact_ids", "explanation", "difficulty_tags",
]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_dicts(path: str | Path, expected_cols: list[str]) -> list[dict]:
    # newline="" + utf-8 so the csv module handles quoted commas/newlines.
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != expected_cols:
            raise ValueError(
                f"{path}: unexpected columns {reader.fieldnames}, "
                f"expected {expected_cols}"
            )
        return list(reader)


def load_answers(path: str | Path | None = None, settings: Settings | None = None) -> list[Answer]:
    settings = settings or get_settings()
    path = path or settings.answers_csv
    out = []
    for r in _read_dicts(path, ANSWER_COLS):
        out.append(Answer(
            answer_id=r["answer_id"],
            split=r["split"],
            brand_id=r["brand_id"],
            question=r["question"],
            ai_answer=r["ai_answer"],  # verbatim
        ))
    return out


def load_brand_facts(path: str | Path | None = None, settings: Settings | None = None) -> list[BrandFact]:
    settings = settings or get_settings()
    path = path or settings.brand_facts_csv
    out = []
    for r in _read_dicts(path, FACT_COLS):
        out.append(BrandFact(
            fact_id=r["fact_id"], brand_id=r["brand_id"], model_id=r["model_id"],
            field=r["field"], value=r["value"], value_type=r["value_type"],
            unit=r["unit"], condition=r["condition"],
            valid_from=r["valid_from"], valid_to=r["valid_to"],
        ))
    return out


def load_gold(path: str | Path | None = None, settings: Settings | None = None) -> list[GoldClaim]:
    """Gold claims. ONLY to be called from the evaluation/reporting pipeline."""
    settings = settings or get_settings()
    path = path or settings.development_gold_csv
    out = []
    for r in _read_dicts(path, GOLD_COLS):
        out.append(GoldClaim(
            answer_id=r["answer_id"],
            claim_id=r["claim_id"],
            start_char=int(r["start_char"]),
            end_char=int(r["end_char"]),
            quote=r["quote"],
            expected_label=r["expected_label"],
            evidence_fact_ids=[x for x in r["evidence_fact_ids"].split("|") if x],
            explanation=r["explanation"],
            difficulty_tags=[x for x in r["difficulty_tags"].split("|") if x],
        ))
    return out


def answers_for_split(split: str, settings: Settings | None = None) -> list[Answer]:
    return [a for a in load_answers(settings=settings) if a.split == split]


def source_hashes(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    return {
        "answers.csv": sha256_file(settings.answers_csv),
        "brand_facts.csv": sha256_file(settings.brand_facts_csv),
        "development_gold.csv": sha256_file(settings.development_gold_csv),
    }


def interpret_value(fact: BrandFact):
    """Interpret value by declared value_type (assessment rule).

    boolean text is literally 'true'/'false' — never bool('false').
    """
    v = fact.value
    if fact.value_type == "boolean":
        return {"true": True, "false": False}.get(v.strip().lower())
    if fact.value_type == "number":
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return None
    return v
