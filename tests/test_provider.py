from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claimlens.config import Settings
from claimlens.provider import OpenAIProvider, ProviderError


def test_malformed_sdk_response_wrapped_without_leaking():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.settings = Settings()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **k: SimpleNamespace(choices=[]))))
    with pytest.raises(ProviderError, match="Malformed provider response"):
        provider.chat([], [])


def test_sdk_retries_disabled_and_timeout_explicit(monkeypatch):
    import openai
    constructor = Mock()
    monkeypatch.setattr(openai, "OpenAI", constructor)
    OpenAIProvider(Settings(openai_api_key="synthetic", request_timeout_s=7))
    assert constructor.call_args.kwargs["timeout"] == 7
    assert constructor.call_args.kwargs["max_retries"] == 0
