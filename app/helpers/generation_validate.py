from __future__ import annotations

import re
from typing import Any

from app.services.generation.models import GenerationResult, ungrounded_fallback

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


def validate_generation_payload(
    payload: dict[str, Any] | None,
    allowed_source_ids: set[str],
) -> GenerationResult | None:
    if not isinstance(payload, dict):
        return None
    if "grounded" not in payload or "answer" not in payload or "source_ids" not in payload:
        return None
    if not isinstance(payload["grounded"], bool):
        return None
    answer = payload["answer"]
    if not isinstance(answer, str) or not answer.strip():
        return None
    source_ids = payload["source_ids"]
    if not isinstance(source_ids, list) or any(not isinstance(item, str) or not item for item in source_ids):
        return None
    if not payload["grounded"]:
        return ungrounded_fallback()
    unique_ids = unique_preserve_order([canonical_source_id(item) for item in source_ids])
    unique_ids = [item for item in unique_ids if item]
    if not unique_ids:
        return None
    if not set(unique_ids) <= allowed_source_ids:
        return None
    return GenerationResult(grounded=True, answer=answer.strip(), source_ids=unique_ids)


def explain_generation_payload(
    payload: dict[str, Any] | None,
    allowed_source_ids: set[str],
) -> dict[str, Any]:
    """Observe grounding validation. Does not change validate_generation_payload."""
    available = sorted(allowed_source_ids)
    base = {
        "grounded_requested": True,
        "grounded_returned": None,
        "validator_result": False,
        "validator_reason": "",
        "returned_source_ids": [],
        "valid_source_ids": [],
        "invalid_source_ids": [],
        "available_source_ids": available,
        "source_id_validation_passed": False,
        "fallback_triggered": True,
    }
    if not isinstance(payload, dict):
        base["validator_reason"] = "invalid_json"
        return base
    grounded = payload.get("grounded")
    base["grounded_returned"] = grounded if isinstance(grounded, bool) else None
    if "grounded" not in payload or "answer" not in payload or "source_ids" not in payload:
        base["validator_reason"] = "missing_fields"
        return base
    if not isinstance(payload.get("grounded"), bool):
        base["validator_reason"] = "invalid_grounded_flag"
        return base
    if not isinstance(payload.get("answer"), str) or not str(payload.get("answer") or "").strip():
        base["validator_reason"] = "invalid_answer"
        return base
    source_ids = payload.get("source_ids")
    if not isinstance(source_ids, list) or any(not isinstance(item, str) or not item for item in source_ids):
        base["validator_reason"] = "invalid_source_ids"
        return base
    if not payload["grounded"]:
        unique_ids = unique_preserve_order([canonical_source_id(item) for item in source_ids])
        unique_ids = [item for item in unique_ids if item]
        base["returned_source_ids"] = unique_ids
        base["valid_source_ids"] = [item for item in unique_ids if item in allowed_source_ids]
        base["invalid_source_ids"] = [item for item in unique_ids if item not in allowed_source_ids]
        base["validator_result"] = True
        base["validator_reason"] = "llm_ungrounded"
        base["fallback_triggered"] = True
        return base
    unique_ids = unique_preserve_order([canonical_source_id(item) for item in source_ids])
    unique_ids = [item for item in unique_ids if item]
    base["returned_source_ids"] = unique_ids
    base["valid_source_ids"] = [item for item in unique_ids if item in allowed_source_ids]
    base["invalid_source_ids"] = [item for item in unique_ids if item not in allowed_source_ids]
    if not unique_ids:
        base["validator_reason"] = "empty_source_ids"
        return base
    if not set(unique_ids) <= allowed_source_ids:
        base["validator_reason"] = "source_id_not_found"
        return base
    base["validator_result"] = True
    base["source_id_validation_passed"] = True
    base["fallback_triggered"] = False
    base["validator_reason"] = "ok"
    return base
