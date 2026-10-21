from claimlens.config import Settings
from claimlens.security import redact
from claimlens.provider import ToolCall, ProviderError
import pytest


def test_settings_omit_all_database_credentials():
    settings = Settings(openai_api_key="DO_NOT_LEAK", database_url="postgresql://u:password@localhost/db?sslpassword=also_secret")
    assert "DO_NOT_LEAK" not in str(settings.masked())
    assert settings.masked()["database_url"] == "[redacted]"


def test_trace_redaction_recursive():
    raw = {"arguments": ["DO_NOT_LEAK", "postgresql://u:secret@localhost/db", "sk-abcdefghijklmnop"]}
    safe = str(redact(raw, ["DO_NOT_LEAK"]))
    assert "DO_NOT_LEAK" not in safe and "secret" not in safe and "abcdefghijklmnop" not in safe


def test_duplicate_json_keys_rejected():
    with pytest.raises(ProviderError):
        ToolCall("x", "get_brand_facts", '{"brand_id":"cleartrail","brand_id":"stonewave"}').parsed_args()
