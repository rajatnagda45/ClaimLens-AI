"""Read-only `get_brand_facts(brand_id)` tool.

The ONLY factual authority is brand_facts.csv at fact version
`speaker-catalogue-v4`. The tool is allowlisted by name and validates its
brand argument strictly. Unknown brands and malformed requests are errors.
Nothing here ever executes arbitrary names or text from answers.
"""
from __future__ import annotations

from dataclasses import asdict

from . import FACT_VERSION
from .data_loader import BrandFact, load_brand_facts, interpret_value

TOOL_NAME = "get_brand_facts"


class ToolError(Exception):
    """Raised for unknown brands or malformed tool arguments."""


class FactTool:
    """Encapsulates the catalogue and the single read-only lookup."""

    def __init__(self, facts: list[BrandFact] | None = None, *, settings=None):
        self._facts = facts if facts is not None else load_brand_facts(settings=settings)
        self._brands = {f.brand_id for f in self._facts}

    @property
    def known_brands(self) -> set[str]:
        return set(self._brands)

    def get_brand_facts(self, brand_id: str) -> dict:
        """Return catalogue rows for `brand_id` at fact_version v4.

        Comparisons may reference another known brand, so ANY known brand is
        a permitted lookup — the primary brand is context, not a hard limit.
        """
        if not isinstance(brand_id, str) or not brand_id.strip():
            raise ToolError("brand_id must be a non-empty string")
        brand_id = brand_id.strip()
        if brand_id not in self._brands:
            raise ToolError(f"unknown brand: {brand_id!r}")

        rows = [{**asdict(f), "value": interpret_value(f)} for f in self._facts if f.brand_id == brand_id]
        return {
            "brand_id": brand_id,
            "fact_version": FACT_VERSION,
            "facts": rows,
        }

    @property
    def valid_fact_ids(self) -> set[str]:
        return {f.fact_id for f in self._facts}


# OpenAI tool schema (function-calling definition). Static allowlist.
OPENAI_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Return the authoritative catalogue facts for a known speaker "
                "brand (fact version speaker-catalogue-v4). Call this to look "
                "up any brand mentioned in the answer, including a comparison "
                "brand different from the primary one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_id": {
                        "type": "string",
                        "description": "Brand identifier, e.g. 'cleartrail'.",
                    }
                },
                "required": ["brand_id"],
                "additionalProperties": False,
            },
        },
    }
]

ALLOWED_TOOL_NAMES = {TOOL_NAME}
