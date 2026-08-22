from __future__ import annotations

import re
from typing import Any

PHONE_RE = re.compile(r"(?<![A-Za-z0-9])(\+?\d[\d\s\-()]{8,}\d)(?![A-Za-z0-9])")
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{8,}\b")
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")
AUTH_HEADER_RE = re.compile(r"(?i)(authorization\s*:\s*)\S+")
PASSWORD_ASSIGN_RE = re.compile(r"(?i)(password\s*[:=]\s*)\S+")
TOKEN_KEYS = frozenset(
    {
        "phone",
        "email",
        "user_name",
        "name",
        "api_key",
        "authorization",
        "token",
        "password",
        "secret",
        "openai_api_key",
        "generation_api_key",
    }
)


def redact_text(value: str | None) -> str:
    text = value or ""
    text = PHONE_RE.sub("[phone]", text)
    text = EMAIL_RE.sub("[email]", text)
    text = API_KEY_RE.sub("[api_key]", text)
    text = BEARER_RE.sub("[token]", text)
    text = AUTH_HEADER_RE.sub(r"\1[redacted]", text)
    text = PASSWORD_ASSIGN_RE.sub(r"\1[redacted]", text)
    return text


def redact_value(key: str, value: Any) -> Any:
    lowered = (key or "").lower()
    if lowered in TOKEN_KEYS and value:
        return "[redacted]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_value(key, item) for item in value]
    return value


def redact_mapping(payload: dict[str, Any] | None) -> dict[str, Any]:
    return {str(key): redact_value(str(key), value) for key, value in (payload or {}).items()}
