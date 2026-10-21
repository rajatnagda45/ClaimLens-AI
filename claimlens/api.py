"""Thin FastAPI routes. Business logic lives in agent/runner/services.

Status mapping (assessment):
  /classify         200 ok | 422 bad input/gold fields | 404 unknown brand
                    502 provider/illegal-tool/invalid-output | 504 timeout
  /evaluation-runs  201 created (completed run may contain answer errors)
  /evaluation-runs/{id}  200 | 404 missing | 422 invalid UUID
"""
from __future__ import annotations

import uuid

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse

from .agent import Agent
from .database import AnswerRow, ClaimRow, Run, get_session_factory, init_db
from .fact_tool import FactTool
from .provider import OpenAIProvider, ProviderError, ProviderTimeout
from .runner import Runner, RunFailure
from .schemas import (
    AnswerStatus, ClassifyRequest, ClassifyResponse, ErrorType,
    EvaluationRunRequest,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ClaimLens-AI", version="0.1.0", lifespan=lifespan)

_ERROR_STATUS = {
    ErrorType.PROVIDER_TIMEOUT: 504,
    ErrorType.PROVIDER_ERROR: 502,
    ErrorType.INVALID_OUTPUT: 502,
    ErrorType.ILLEGAL_TOOL: 502,
    ErrorType.TURN_EXHAUSTED: 502,
    ErrorType.VALIDATION_FAILED: 502,
    ErrorType.UNKNOWN_BRAND: 502,
}


def get_fact_tool() -> FactTool:
    return FactTool()


def get_provider():
    try:
        return OpenAIProvider()
    except ProviderTimeout:
        raise HTTPException(504, detail={"error_type": "provider_timeout", "message": "Provider setup timed out"}) from None
    except Exception:
        raise HTTPException(502, detail={"error_type": "provider_error", "message": "Provider setup failed"}) from None


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest, fact_tool: FactTool = Depends(get_fact_tool)):
    # Unknown PRIMARY brand -> 404 before hitting the provider.
    if req.brand_id not in fact_tool.known_brands:
        raise HTTPException(status_code=404, detail=f"unknown brand: {req.brand_id}")

    agent = Agent(get_provider(), fact_tool=fact_tool)
    result = agent.run(
        req.answer_id, req.brand_id, req.question, req.ai_answer, req.agent_version,
    )
    if result.status == AnswerStatus.ERROR:
        status = _ERROR_STATUS.get(result.error_type, 502)
        return JSONResponse(
            status_code=status,
            content={"error_type": result.error_type.value,
                     "message": result.error_message},
        )
    return ClassifyResponse(
        answer_id=result.answer_id, agent_version=req.agent_version,
        claims=result.claims,
    )


@app.post("/evaluation-runs", status_code=201)
def create_evaluation_run(req: EvaluationRunRequest):
    runner = Runner(get_provider())
    try:
        report = runner.run_split(req.agent_version, req.split, persist=True)
    except RunFailure as exc:
        return JSONResponse(status_code=503, content={"run_id": exc.run_id,
                            "status": "failed", "message": str(exc)})
    m = report["metrics"]
    return {
        "run_id": report["metadata"]["run_id"],
        "status": report["metadata"]["status"],
        "split": req.split,
        "agent_version": req.agent_version,
        "counts": {
            "success": m["answers_success"],
            "empty": m["answers_empty"],
            "error": m["answers_error"],
        },
        "metrics": m,
    }


@app.get("/evaluation-runs/{run_id}")
def get_evaluation_run(run_id: str = Path(...)):
    try:
        rid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid UUID")

    Session = get_session_factory()
    with Session() as s:
        run = s.get(Run, rid)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        answers = (
            s.query(AnswerRow)
            .filter(AnswerRow.run_id == rid)
            .order_by(AnswerRow.answer_id)
            .all()
        )
        claims = (
            s.query(ClaimRow)
            .filter(ClaimRow.run_id == rid)
            .order_by(ClaimRow.answer_id, ClaimRow.start_char)
            .all()
        )
        claims_by_answer: dict[str, list] = {}
        for c in claims:
            claims_by_answer.setdefault(c.answer_id, []).append({
                "claim_id": c.claim_id, "start_char": c.start_char,
                "end_char": c.end_char, "quote": c.quote, "label": c.label,
                "fact_ids": c.fact_ids, "reason": c.reason,
                "match_info": c.match_info,
            })
        return {
            "run_id": str(run.id),
            "split": run.split,
            "agent_version": run.agent_version,
            "status": run.status,
            "model": run.model,
            "provider": run.provider,
            "prompt_version": run.prompt_version,
            "dataset_version": run.dataset_version,
            "evaluator_version": run.evaluator_version,
            "fact_version": run.fact_version,
            "settings": run.settings,
            "source_revision": run.source_revision,
            "timestamp": run.created_at.isoformat(),
            "metrics": run.metrics,
            "source_hashes": run.source_hashes,
            "answers": [
                {
                    "answer_id": a.answer_id, "status": a.status,
                    "error_type": a.error_type, "error_message": a.error_message,
                    "latency_ms": a.latency_ms, "turns_used": a.turns_used,
                    "trace": a.trace,
                    "claims": claims_by_answer.get(a.answer_id, []),
                }
                for a in answers
            ],
        }
