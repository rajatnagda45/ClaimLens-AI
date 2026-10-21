"""Live model integration (OpenAI) behind a small provider interface.

The agent loop is provider-agnostic: it speaks in normalized `AssistantTurn`
objects. `OpenAIProvider` talks to the live API with a configurable per-request
timeout. `FakeProvider` (in tests) scripts turns deterministically — fakes are
for tests only and never substitute for the four live runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings, get_settings


class ProviderError(Exception):
    """Non-timeout provider failure (network, 5xx, malformed response)."""


class ProviderTimeout(Exception):
    """Provider request exceeded the configured timeout."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as returned by the model

    def parsed_args(self) -> dict:
        try:
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ProviderError("duplicate JSON argument key")
                    result[key] = value
                return result
            v = json.loads(self.arguments, object_pairs_hook=unique_object)
        except json.JSONDecodeError as e:
            raise ProviderError("tool arguments not valid JSON") from e
        if not isinstance(v, dict):
            raise ProviderError("tool arguments must be a JSON object")
        return v


@dataclass
class AssistantTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_usage: dict | None = None


class Provider(Protocol):
    def chat(
        self, messages: list[dict], tools: list[dict], tool_choice="required",
    ) -> AssistantTurn: ...


class OpenAIProvider:
    def __init__(self, settings: Settings | None = None):
        from openai import OpenAI  # local import keeps module import cheap

        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.request_timeout_s,
            max_retries=0,
        )

    def chat(
        self, messages: list[dict], tools: list[dict], tool_choice="required",
    ) -> AssistantTurn:
        from openai import APITimeoutError, APIError

        try:
            resp = self._client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.settings.temperature,
                timeout=self.settings.request_timeout_s,
            )
        except APITimeoutError as e:
            raise ProviderTimeout("Provider request timed out") from e
        except APIError as e:
            raise ProviderError("Provider request failed") from e
        except Exception as e:  # defensive: any transport issue
            raise ProviderError("Provider request failed") from e

        try:
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = []
            for tc in (msg.tool_calls or []):
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                ))
            usage = None
            if getattr(resp, "usage", None) is not None:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
            return AssistantTurn(content=msg.content, tool_calls=tool_calls, raw_usage=usage)
        except Exception as e:
            raise ProviderError("Malformed provider response") from e
