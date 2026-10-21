"""Best-effort redaction for retained model traces; errors use fixed messages."""
import re


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if not isinstance(value, str):
        return value
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    value = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[redacted]", value)
    value = re.sub(r"(?i)(?:postgres(?:ql)?(?:\+psycopg)?://)[^\s\"']+", "[redacted database URL]", value)
    return value
