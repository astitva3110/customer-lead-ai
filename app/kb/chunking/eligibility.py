"""Chunk embedding eligibility classification for production gate."""

from __future__ import annotations

import re
from enum import StrEnum

from app.kb.chunking.duplicates import (
    DuplicateClassification,
    classify_duplicate_pair,
    duplicate_content_key,
)
from app.kb.chunking.models import ChunkRecord, SplitMethod
from app.kb.chunking.noise import classify_chunk_noise, is_meaningful_small_chunk, should_suppress_noise
from app.kb.enums import ExtractionMethod, SourceType

ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9\s&,\-./():'\"]{2,60}$")
PAGE_PIPE = re.compile(r"^Page\s*\|\s*\d+$", re.I)
PAGE_COUNTER = re.compile(r"^\d+\s*/\s*of\s*\d+$", re.I)
BOILERPLATE = re.compile(
    r"(left blank|by order of the board|all rights reserved|make in india|www\.earkart\.in\s*\|)",
    re.I,
)
PIPELINE_META = re.compile(r"sha256:[a-f0-9]{64}", re.I)
UI_NAV = re.compile(r"(skip to product information|you may also like|open media \d+ in modal)", re.I)
ORPHAN_TIME = re.compile(r"^within (?:seven|ten|\d+) (?:calendar )?days\.?$", re.I)
CLAUSE_ID = re.compile(r"^\d+(?:\.\d+)+\s+")
OCR_GARBAGE = re.compile(r"^[#,\-;]{2,}|^[a-z]{1,2}$|^\W+$")


class ChunkEligibilityStatus(StrEnum):
    EMBED_READY = "EMBED_READY"
    REVIEW = "REVIEW"
    DO_NOT_EMBED = "DO_NOT_EMBED"


class ReviewCategory(StrEnum):
    OCR_UNCERTAINTY = "OCR_UNCERTAINTY"
    SMALL_AMBIGUOUS = "SMALL_AMBIGUOUS"
    HARD_FALLBACK = "HARD_FALLBACK"
    CONTEXT_UNCERTAIN = "CONTEXT_UNCERTAIN"
    OTHER = "OTHER"


def _body(content: str) -> str:
    return content.split("\n\n")[-1].strip() if content else ""


def _is_heading_only(chunk: ChunkRecord) -> bool:
    body = _body(chunk.content)
    path_tail = chunk.section_path[-1] if chunk.section_path else ""
    if body == path_tail and len(body) < 50:
        return True
    return len(body) < 40 and bool(ALL_CAPS_HEADING.match(body)) and chunk.token_count < 8


def _has_section_context(chunk: ChunkRecord) -> bool:
    if len(chunk.section_path) >= 2:
        return True
    if "\n\n" in chunk.content and " > " in chunk.content.split("\n\n")[0]:
        return True
    if CLAUSE_ID.match(_body(chunk.content)):
        return True
    return False


def _ocr_suspicious(body: str) -> bool:
    if OCR_GARBAGE.match(body.strip()):
        return True
    alpha = sum(ch.isalpha() for ch in body)
    return len(body) > 20 and alpha / max(1, len(body)) < 0.5


def _review_category(reasons: list[str], flags: list[str]) -> ReviewCategory:
    if any("ocr" in r for r in reasons) or any("ocr" in f for f in flags):
        return ReviewCategory.OCR_UNCERTAINTY
    if any("small" in r for r in reasons) or "small_ambiguous" in flags:
        return ReviewCategory.SMALL_AMBIGUOUS
    if any("hard_fallback" in r for r in reasons) or "hard_fallback_short" in flags:
        return ReviewCategory.HARD_FALLBACK
    if any("context" in r or "insufficient" in r for r in reasons) or "orphan_time_limit" in flags:
        return ReviewCategory.CONTEXT_UNCERTAIN
    return ReviewCategory.OTHER


def classify_chunk_eligibility(
    chunk: ChunkRecord,
    *,
    seen_strict: dict[str, ChunkRecord],
) -> tuple[ChunkEligibilityStatus, list[str], list[str], ReviewCategory | None]:
    """Return eligibility status, reasons, flags, and optional review category."""
    flags: list[str] = []
    reasons: list[str] = []

    if not chunk.content.strip():
        flags.append("empty_content")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["empty_content"], flags, None

    if chunk.token_count > 512:
        flags.append("token_violation")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["token_limit_exceeded"], flags, None

    if not chunk.document_id or not chunk.source_url or not chunk.canonical_url:
        flags.append("provenance_failure")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["missing_provenance"], flags, None

    if PIPELINE_META.search(chunk.content):
        flags.append("pipeline_metadata")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["pipeline_metadata_leak"], flags, None

    body = _body(chunk.content)

    noise = classify_chunk_noise(chunk.content, split_method=chunk.split_method, token_count=chunk.token_count)
    if should_suppress_noise(noise):
        flags.append(f"noise_{noise.reason}")
        return ChunkEligibilityStatus.DO_NOT_EMBED, [f"noise:{noise.category}"], flags, None

    if PAGE_PIPE.match(body) or PAGE_COUNTER.match(body):
        flags.append("page_counter")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["page_counter"], flags, None

    if _is_heading_only(chunk):
        flags.append("isolated_heading")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["heading_only"], flags, None

    strict_key = duplicate_content_key(chunk)
    if strict_key in seen_strict:
        dup = classify_duplicate_pair(seen_strict[strict_key], chunk)
        if dup in {DuplicateClassification.SAME_CONTEXT_DUPLICATE, DuplicateClassification.BOILERPLATE}:
            flags.append("duplicate_same_context")
            return ChunkEligibilityStatus.DO_NOT_EMBED, [f"duplicate:{dup.value}"], flags, None
    seen_strict[strict_key] = chunk

    if BOILERPLATE.search(chunk.content) and len(body) < 120:
        flags.append("boilerplate")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["boilerplate"], flags, None

    if UI_NAV.search(body) and chunk.token_count < 25:
        flags.append("ui_navigation")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["navigation_ui"], flags, None

    if chunk.token_count <= 3 and not is_meaningful_small_chunk(body):
        flags.append("extremely_low_information")
        return ChunkEligibilityStatus.DO_NOT_EMBED, ["tiny_meaningless_fragment"], flags, None

    if ORPHAN_TIME.match(body) and not _has_section_context(chunk):
        flags.append("orphan_time_limit")
        reasons.append("insufficient_context")
        return ChunkEligibilityStatus.REVIEW, reasons, flags, _review_category(reasons, flags)

    if chunk.extraction_method == ExtractionMethod.OCR and _ocr_suspicious(body):
        flags.append("ocr_suspicious")
        reasons.append("ocr_quality_uncertain")
        return ChunkEligibilityStatus.REVIEW, reasons, flags, _review_category(reasons, flags)

    if chunk.token_count < 8 and not is_meaningful_small_chunk(body):
        flags.append("small_ambiguous")
        reasons.append("small_chunk_needs_review")
        return ChunkEligibilityStatus.REVIEW, reasons, flags, _review_category(reasons, flags)

    if chunk.split_method == SplitMethod.HARD_TOKEN_FALLBACK and chunk.token_count < 40:
        flags.append("hard_fallback_short")
        reasons.append("hard_fallback_boundary")
        return ChunkEligibilityStatus.REVIEW, reasons, flags, _review_category(reasons, flags)

    if chunk.source_type in {SourceType.PDF, SourceType.SCANNED_PDF} and chunk.page_number is None:
        if chunk.extraction_method == ExtractionMethod.OCR:
            flags.append("missing_page_provenance")
            reasons.append("ocr_missing_page")
            return ChunkEligibilityStatus.REVIEW, reasons, flags, _review_category(reasons, flags)

    if "open media" in chunk.content.lower() and chunk.token_count < 100:
        flags.append("shopify_remnant")
        reasons.append("embedded_ui_remnant")
        return ChunkEligibilityStatus.REVIEW, reasons, flags, _review_category(reasons, flags)

    return ChunkEligibilityStatus.EMBED_READY, [], flags, None
