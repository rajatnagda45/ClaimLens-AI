"""Shared test fixtures and a deterministic FakeProvider.

Fakes are for controlled tests ONLY and never substitute for the four live
runs. The FakeProvider records every message it receives so tests can assert
that gold data never enters the model/tool payload.
"""
from __future__ import annotations

import json

import pytest

from claimlens.provider import AssistantTurn, ProviderError, ProviderTimeout, ToolCall


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
