"""Pydantic request, response, claim, and error schemas.

These schemas are the single source of truth for shape and validation of
agent I/O, API I/O, and persisted records. Client-supplied gold/claim fields
are explicitly forbidden on the API input (see `ClassifyRequest`).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Label = Literal["supported", "contradicted", "unknown"]
AgentVersion = Literal["v1", "v2"]
Split = Literal["development", "check"]


class Claim(BaseModel):
    """A single validated, span-located factual claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str
    label: Label
    fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class ModelClaim(BaseModel):
    """Raw claim as returned by the model: exact quote + label + evidence.

    The model proposes a quote; Python independently locates the offsets.
    We deliberately do NOT trust model-provided character counts.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    quote: str
    label: Label
    fact_ids: list[str]
    reason: str


class ModelExtraction(BaseModel):
    """Top-level structured output expected from the model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[ModelClaim]


class AnswerStatus(str, Enum):
    SUCCESS = "success"          # validated predictions (may be empty list)
    EMPTY = "empty"              # validated, zero claims
    ERROR = "error"              # answer-level error, no accepted predictions


class ErrorType(str, Enum):
    PROVIDER_ERROR = "provider_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    INVALID_OUTPUT = "invalid_output"
    ILLEGAL_TOOL = "illegal_tool"
    UNKNOWN_BRAND = "unknown_brand"
    TURN_EXHAUSTED = "turn_exhausted"
    VALIDATION_FAILED = "validation_failed"


class ClassifyRequest(BaseModel):
    """Input to POST /classify and the agent.

    `extra="forbid"` rejects any client-supplied gold span/label/claim fields
    with a 422, satisfying the assessment's prompt-injection / data-leak rule.
    """

    model_config = ConfigDict(extra="forbid")

    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: AgentVersion = "v2"


class AnswerResult(BaseModel):
    """Terminal outcome for one scheduled answer."""

    answer_id: str
    status: AnswerStatus
    claims: list[Claim] = Field(default_factory=list)
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    turns_used: Optional[int] = None
    trace: Optional[list[dict]] = None  # sanitized model/tool exchange


class ClassifyResponse(BaseModel):
    answer_id: str
    agent_version: AgentVersion
    claims: list[Claim]


class ErrorResponse(BaseModel):
    error_type: str
    message: str


# ---- Evaluation metrics ----

class Metrics(BaseModel):
    """Deterministic evaluation metrics. Null on empty / check splits."""

    gold_claims: Optional[int] = None          # G
    accepted_predictions: Optional[int] = None  # P
    span_matches: Optional[int] = None          # M
    correct_label_matches: Optional[int] = None  # C

    extraction_precision: Optional[float] = None
    extraction_recall: Optional[float] = None
    extraction_f1: Optional[float] = None

    joint_precision: Optional[float] = None
    joint_recall: Optional[float] = None
    joint_f1: Optional[float] = None

    gold_coverage: Optional[float] = None  # C/G

    missed_gold_ids: Optional[list[str]] = None
    unmatched_prediction_ids: Optional[list[str]] = None
    wrong_label_ids: Optional[list[str]] = None

    answers_success: int = 0
    answers_empty: int = 0
    answers_error: int = 0


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: AgentVersion
    split: Split
