from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_markdown_fence(text: str) -> str:
    stripped = (text or "").strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = strip_markdown_fence(strip_think_blocks(text))
    parsed = _load_object(stripped)
    if parsed is not None:
        return parsed
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    return _load_object(stripped[start : end + 1])


def _load_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
