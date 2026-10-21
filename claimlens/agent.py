"""Agent: real tool-calling loop for extraction + classification.

Flow per answer (<= max_turns model turns):
  1. Send system prompt (v1/v2) + the untrusted answer as data.
  2. The model may call get_brand_facts(brand_id) one or more times per turn.
  3. The model finishes by calling submit_claims(...) with exact quotes+labels.
  4. Python locates offsets and validates every invariant (validator.py).

Guarantees:
- Tool names are checked against an explicit allowlist. Unknown tool name =>
  ILLEGAL_TOOL answer-level error. Arbitrary names are never dispatched.
- Unknown requested brand => UNKNOWN_BRAND error.
- Turn budget exhausted without submit_claims => TURN_EXHAUSTED error.
- Provider timeout/error/malformed output => explicit answer-level error.
- A provider/validation failure is NEVER downgraded to an `unknown` claim and
  NEVER yields partially-accepted predictions.
- The sanitized model/tool exchange is captured as evidence (no secrets).
"""
from __future__ import annotations

import json
import time

from .config import Settings, get_settings
from .fact_tool import ALLOWED_TOOL_NAMES, OPENAI_TOOL_SPEC, FactTool, ToolError
from .prompts import get_system_prompt
from .provider import (
    AssistantTurn, Provider, ProviderError, ProviderTimeout, ToolCall,
)
from .schemas import (
    AnswerResult, AnswerStatus, ErrorType, ModelClaim, ModelExtraction,
)
from .validator import ValidationError, build_claims

SUBMIT_TOOL = "submit_claims"

SUBMIT_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": SUBMIT_TOOL,
        "description": (
            "Submit the final list of extracted, classified claims. Call this "
            "exactly once when finished."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string"},
                            "label": {
                                "type": "string",
                                "enum": ["supported", "contradicted", "unknown"],
                            },
                            "fact_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["quote", "label", "fact_ids", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
    },
}

TOOLS = OPENAI_TOOL_SPEC + [SUBMIT_TOOL_SPEC]


class _AnswerError(Exception):
    def __init__(self, etype: ErrorType, message: str):
        self.etype = etype
        self.message = message
        super().__init__(message)


# Errors the model can plausibly fix given one more turn (bad quote/label/fact).
_RECOVERABLE = {ErrorType.VALIDATION_FAILED, ErrorType.INVALID_OUTPUT}


def _sanitize_facts_for_trace(result: dict) -> dict:
    return {"brand_id": result["brand_id"], "fact_count": len(result["facts"])}


class Agent:
    def __init__(
        self,
        provider: Provider,
        fact_tool: FactTool | None = None,
        settings: Settings | None = None,
    ):
        self.provider = provider
        self.settings = settings or get_settings()
        self.fact_tool = fact_tool or FactTool()

    def run(
        self, answer_id: str, brand_id: str, question: str, ai_answer: str,
        agent_version: str,
    ) -> AnswerResult:
        """Process one answer into a terminal AnswerResult (never raises)."""
        start = time.perf_counter()
        trace: list[dict] = []
        turns_used = 0
        try:
            claims = self._loop(
                answer_id, brand_id, question, ai_answer, agent_version, trace,
            )
            latency = (time.perf_counter() - start) * 1000
            status = AnswerStatus.EMPTY if not claims else AnswerStatus.SUCCESS
            return AnswerResult(
                answer_id=answer_id, status=status, claims=claims,
                latency_ms=latency, turns_used=trace_turns(trace), trace=trace,
            )
        except _AnswerError as e:
            latency = (time.perf_counter() - start) * 1000
            return AnswerResult(
                answer_id=answer_id, status=AnswerStatus.ERROR,
                error_type=e.etype, error_message=e.message,
                latency_ms=latency, turns_used=trace_turns(trace), trace=trace,
            )

    # ---- internal ----

    def _loop(self, answer_id, brand_id, question, ai_answer, agent_version, trace):
        system = get_system_prompt(agent_version)
        known = ", ".join(sorted(self.fact_tool.known_brands))
        user = (
            f"answer_id: {answer_id}\n"
            f"primary brand_id: {brand_id}\n"
            f"Valid brand_id values for get_brand_facts (use EXACTLY one of "
            f"these; a model name like 'mini'/'plus' is NOT a brand_id): {known}\n"
            f"question (context only, do NOT extract from it): {question}\n\n"
            f"--- BEGIN UNTRUSTED ANSWER TEXT ---\n{ai_answer}\n"
            f"--- END UNTRUSTED ANSWER TEXT ---"
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        max_turns = self.settings.max_turns
        for _turn in range(max_turns):
            # On the final allowed turn, force submit_claims so the loop always
            # terminates with structured output rather than prose or a stray
            # extra lookup (otherwise => invalid_output / turn_exhausted).
            is_last = _turn == max_turns - 1
            tool_choice = (
                {"type": "function", "function": {"name": SUBMIT_TOOL}}
                if is_last else "required"
            )
            turn = self._chat(messages, tool_choice)
            trace.append({"role": "assistant", "content": turn.content,
                          "tool_calls": [{"name": tc.name, "arguments": tc.arguments}
                                         for tc in turn.tool_calls]})

            if not turn.tool_calls:
                # Model responded with text but did not use the submit tool.
                # Treat missing structured output as invalid.
                raise _AnswerError(
                    ErrorType.INVALID_OUTPUT,
                    "model did not call submit_claims",
                )

            # Record the assistant tool-call message for the provider protocol.
            messages.append(self._assistant_msg(turn))

            submit_call = None
            for tc in turn.tool_calls:
                if tc.name == SUBMIT_TOOL:
                    submit_call = tc  # handle after fact lookups
                    continue
                if tc.name not in ALLOWED_TOOL_NAMES:
                    raise _AnswerError(
                        ErrorType.ILLEGAL_TOOL, f"illegal tool: {tc.name!r}",
                    )
                tool_result = self._exec_fact_tool(tc)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })
                trace.append({"role": "tool", "name": tc.name,
                              "result": _sanitize_facts_for_trace(tool_result)})

            if submit_call is not None:
                try:
                    return self._finalize(answer_id, ai_answer, submit_call)
                except _AnswerError as e:
                    # Respond to the submit tool call so the history stays
                    # valid, then decide whether a recovery turn is available.
                    messages.append({
                        "role": "tool", "tool_call_id": submit_call.id,
                        "content": json.dumps({"error": e.message}),
                    })
                    trace.append({"role": "recovery", "error": e.message})
                    if is_last or e.etype not in _RECOVERABLE:
                        raise
                    # One explicit, bounded recovery attempt (within turn budget).
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your submit_claims was rejected: " + e.message +
                            " Fix ONLY that problem and call submit_claims again. "
                            "Copy each quote VERBATIM (character-for-character, "
                            "including final punctuation) from the answer text, "
                            "use only labels supported/contradicted/unknown, and "
                            "use only fact_ids that exist in the catalogue."
                        ),
                    })

        raise _AnswerError(
            ErrorType.TURN_EXHAUSTED,
            f"no submit_claims within {self.settings.max_turns} turns",
        )

    def _chat(self, messages, tool_choice="required") -> AssistantTurn:
        try:
            return self.provider.chat(messages, TOOLS, tool_choice)
        except ProviderTimeout as e:
            raise _AnswerError(ErrorType.PROVIDER_TIMEOUT, str(e)) from e
        except ProviderError as e:
            raise _AnswerError(ErrorType.PROVIDER_ERROR, str(e)) from e

    def _exec_fact_tool(self, tc: ToolCall) -> dict:
        try:
            args = tc.parsed_args()
        except ProviderError as e:
            raise _AnswerError(ErrorType.INVALID_OUTPUT, str(e)) from e
        brand = args.get("brand_id")
        if not isinstance(brand, str):
            raise _AnswerError(ErrorType.INVALID_OUTPUT, "brand_id missing/invalid")
        try:
            return self.fact_tool.get_brand_facts(brand)
        except ToolError as e:
            raise _AnswerError(ErrorType.UNKNOWN_BRAND, str(e)) from e

    def _finalize(self, answer_id, ai_answer, submit_call: ToolCall):
        try:
            args = submit_call.parsed_args()
            extraction = ModelExtraction.model_validate(args)
        except Exception as e:
            raise _AnswerError(
                ErrorType.INVALID_OUTPUT, f"bad submit_claims payload: {e}",
            ) from e
        model_claims: list[ModelClaim] = extraction.claims
        try:
            claims = build_claims(
                answer_id, ai_answer, model_claims, self.fact_tool.valid_fact_ids,
            )
        except ValidationError as e:
            raise _AnswerError(ErrorType.VALIDATION_FAILED, str(e)) from e
        return claims

    @staticmethod
    def _assistant_msg(turn: AssistantTurn) -> dict:
        return {
            "role": "assistant",
            "content": turn.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in turn.tool_calls
            ],
        }


def trace_turns(trace: list[dict]) -> int:
    return sum(1 for t in trace if t.get("role") == "assistant")
