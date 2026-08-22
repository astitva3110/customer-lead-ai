"""Chunk-level noise detection for OCR/PDF debris and UI fragments."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.kb.chunking.models import SplitMethod

PAGE_COUNTER = re.compile(r"^\d+\s*/\s*of\s*\d+$", re.I)
LIST_NUMERIC_DEBRIS = re.compile(r"^- \d+$")
LIST_SHORT_DEBRIS = re.compile(r"^- [a-z]{1,4}$", re.I)
SHOPIFY_UI = re.compile(r"^open media \d+ in modal$", re.I)
ROMAN_NUMERAL_DEBRIS = re.compile(r"^[IVXLC\d+\s=\-\n]+$", re.I)
CLAUSE = re.compile(r"^\d+(?:\.\d+)+\s+")
MEANINGFUL_SHORT = re.compile(
    r"(battery size|within \w+ days|rma required|return/replacement|eligibility|ms\b|hrs\b|mah\b)",
    re.I,
)

HIGH_CONFIDENCE = 0.85


@dataclass(frozen=True)
class NoiseClassification:
    """Classification for chunk-level noise filtering."""

    category: str | None
    confidence: float
    reason: str


def _body_text(content: str) -> str:
    if "\n\n" in content:
        return content.split("\n\n")[-1].strip()
    return content.strip()


def is_meaningful_small_chunk(body: str) -> bool:
    """Small chunks that represent valid standalone knowledge."""
    if CLAUSE.match(body):
        return True
    if ":" in body and len(body.split()) >= 2:
        return True
    if MEANINGFUL_SHORT.search(body):
        return True
    return False


def classify_chunk_noise(
    content: str,
    *,
    split_method: SplitMethod,
    token_count: int,
) -> NoiseClassification:
    """Detect high-confidence standalone debris; uncertain content is retained."""
    body = _body_text(content)
    if not body:
        return NoiseClassification(None, 0.0, "")

    if is_meaningful_small_chunk(body):
        return NoiseClassification(None, 0.0, "")

    if SHOPIFY_UI.match(body):
        return NoiseClassification("standalone_low_value", 0.98, "shopify_ui_fragment")

    if PAGE_COUNTER.match(body):
        return NoiseClassification("context_noise", 0.95, "page_counter")

    if LIST_NUMERIC_DEBRIS.match(body):
        return NoiseClassification("standalone_low_value", 0.95, "numeric_list_debris")

    if LIST_SHORT_DEBRIS.match(body):
        return NoiseClassification("standalone_low_value", 0.9, "short_list_debris")

    if body in {"-", "i", "g", "(B)"} and token_count <= 2:
        return NoiseClassification("standalone_low_value", 0.95, "single_token_debris")

    if token_count <= 5 and ROMAN_NUMERAL_DEBRIS.match(body) and not CLAUSE.match(body):
        return NoiseClassification("standalone_low_value", 0.88, "ocr_roman_numeral_debris")

    if token_count <= 3 and split_method == SplitMethod.HARD_TOKEN_FALLBACK:
        words = body.split()
        if len(words) <= 4 and not body.endswith("."):
            return NoiseClassification("context_noise", 0.9, "hard_split_orphan_tail")

    return NoiseClassification(None, 0.0, "")


def should_suppress_noise(classification: NoiseClassification) -> bool:
    return classification.category is not None and classification.confidence >= HIGH_CONFIDENCE
