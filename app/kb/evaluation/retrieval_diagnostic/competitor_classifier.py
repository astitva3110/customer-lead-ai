"""Deterministic competitor chunk classification for targeted inspection."""

from __future__ import annotations

from typing import Any


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def classify_competitor(hit: dict[str, Any], expected_chunk_ids: set[str]) -> str:
    if hit.get("chunk_id") in expected_chunk_ids:
        return "CORRECT_KNOWLEDGE"

    text = hit.get("text") or ""
    lowered = text.lower()
    section = " > ".join(hit.get("section_path") or []).lower()
    token_count = hit.get("token_count") or estimate_token_count(text)

    if "![" in text or (".png" in lowered and token_count < 40):
        return "IMAGE_OR_ICON"
    if "you may also like" in lowered:
        return "PRODUCT_CROSS_SELL"
    if any(token in lowered for token in ("customer reviews", "write a review", "sort by", "be the first")):
        return "UI_NOISE"
    if any(token in lowered for token in ("view full details", "quantity", "sold out", "add to cart")):
        return "NAVIGATION"
    if token_count < 25 and len(text.strip()) < 140:
        return "HEADING_ONLY"
    if any(token in lowered for token in ("redefining hearing care", "investor", "prospectus", "annual report")):
        return "GENERAL_MARKETING"
    if any(token in lowered for token in ("hearing aid", "amplifier", "microphone", "speaker", "psap")):
        return "RELATED_KNOWLEDGE"
    if any(token in section for token in ("faq", "hearing aid", "bluup", "tiny", "ven")):
        return "RELATED_KNOWLEDGE"
    return "UNKNOWN"


def targeted_diagnosis(
    *,
    expected_chunk_rank: int | None,
    expected_document_best_rank: int | None,
    expected_chunk_found: bool,
) -> str:
    if not expected_chunk_found:
        return "EXPECTED_CHUNK_NOT_FOUND"
    if expected_chunk_rank is not None and expected_chunk_rank <= 10:
        return "DOCUMENT_AND_CHUNK_RETRIEVED"
    if (
        expected_document_best_rank is not None
        and expected_document_best_rank <= 10
        and (expected_chunk_rank is None or expected_chunk_rank > 10)
    ):
        return "DOCUMENT_RETRIEVED_CHUNK_MISSED"
    if expected_document_best_rank is None or expected_document_best_rank > 20:
        return "DOCUMENT_NOT_RETRIEVED"
    if expected_chunk_rank is None or expected_chunk_rank > 10:
        return "DOCUMENT_RETRIEVED_CHUNK_MISSED"
    return "DOCUMENT_NOT_RETRIEVED"
