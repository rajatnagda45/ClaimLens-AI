"""Separately versioned prompts. v1 = baseline, v2 = one targeted change."""
from __future__ import annotations

from .v1 import SYSTEM_PROMPT as V1_SYSTEM
from .v2 import SYSTEM_PROMPT as V2_SYSTEM

PROMPTS = {"v1": V1_SYSTEM, "v2": V2_SYSTEM}

# Human-readable prompt version strings stored in run metadata.
PROMPT_VERSIONS = {"v1": "extract-classify-v1", "v2": "extract-classify-v2"}


def get_system_prompt(version: str) -> str:
    if version not in PROMPTS:
        raise KeyError(f"unknown prompt version: {version}")
    return PROMPTS[version]
