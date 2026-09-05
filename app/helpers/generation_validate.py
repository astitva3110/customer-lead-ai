from __future__ import annotations

import re
from typing import Any

from app.services.generation.models import GenerationResult

_SOURCE_DISPLAY_RE = re.compile(r"^\[?SOURCE_ID:\s*(.+?)\]?$", re.IGNORECASE)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def canonical_source_id(value: str) -> str:
    """Map prompt display form to the raw chunk_id used internally."""
    text = (value or "").strip()
    match = _SOURCE_DISPLAY_RE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    return text.strip("[]").strip()


def coerce_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    source_ids = out.get("source_ids")
    if isinstance(source_ids, str) and source_ids.strip():
        out["source_ids"] = [source_ids.strip()]
    elif isinstance(source_ids, list):
        out["source_ids"] = [str(item).strip() for item in source_ids if str(item).strip()]
    return out


def resolve_allowed_source_ids(source_ids: list[str], allowed_source_ids: set[str]) -> list[str] | None:
    mapped: list[str] = []
    for item in source_ids:
        canon = canonical_source_id(item)
        if canon in allowed_source_ids:
            mapped.append(canon)
            continue
        matches = [allowed for allowed in allowed_source_ids if allowed.startswith(canon) or canon.startswith(allowed)]
        if len(matches) != 1:
            continue
        mapped.append(matches[0])
    resolved = unique_preserve_order(mapped)
    return resolved or None


def validate_generation_payload(
    payload: dict[str, Any] | None,
    allowed_source_ids: set[str],
) -> GenerationResult | None:
    if not isinstance(payload, dict):
        return None
    payload = coerce_generation_payload(payload)
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    source_ids = payload.get("source_ids")
    unique_ids: list[str] = []
    if isinstance(source_ids, list):
        unique_ids = resolve_allowed_source_ids(source_ids, allowed_source_ids) or []
    if not unique_ids:
        unique_ids = unique_preserve_order(sorted(item for item in allowed_source_ids if item))
    return GenerationResult(grounded=True, answer=answer.strip(), source_ids=unique_ids)


def explain_generation_payload(
    payload: dict[str, Any] | None,
    allowed_source_ids: set[str],
) -> dict[str, Any]:
    available = sorted(allowed_source_ids)
    base = {
        "grounded_requested": False,
        "grounded_returned": None,
        "validator_result": False,
        "validator_reason": "",
        "returned_source_ids": [],
        "valid_source_ids": [],
        "invalid_source_ids": [],
        "available_source_ids": available,
        "source_id_validation_passed": True,
        "fallback_triggered": True,
    }
    if not isinstance(payload, dict):
        base["validator_reason"] = "invalid_json"
        return base
    payload = coerce_generation_payload(payload)
    if not isinstance(payload.get("answer"), str) or not str(payload.get("answer") or "").strip():
        base["validator_reason"] = "invalid_answer"
        return base
    source_ids = payload.get("source_ids")
    returned: list[str] = []
    if isinstance(source_ids, list):
        returned = unique_preserve_order([canonical_source_id(item) for item in source_ids if str(item).strip()])
    unique_ids = resolve_allowed_source_ids(returned, allowed_source_ids) or []
    base["returned_source_ids"] = returned
    base["valid_source_ids"] = unique_ids
    base["invalid_source_ids"] = [item for item in returned if item not in unique_ids]
    base["grounded_returned"] = True
    base["validator_result"] = True
    base["fallback_triggered"] = False
    base["validator_reason"] = "ok"
    return base
