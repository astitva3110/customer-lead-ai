from dataclasses import dataclass, field
import re

from app.kb.enums import ExclusionReason, ProcessingStatus, SourceType
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.raw import CleanedArtifact, RawArtifact
from app.kb.policy.domain_policy import detect_exclusion

# Compare cleaned output against de-boilerplated raw baseline, not raw including nav.
MIN_BASELINE_RETENTION_RATIO = 0.50
MIN_CLEANED_MEANINGFUL_CHARS = 80
MIN_SHORT_PDF_BASELINE_CHARS = 150


@dataclass
class ValidationResult:
    passed: bool
    status: ProcessingStatus
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    excluded: bool = False
    exclusion_reason: ExclusionReason | None = None


def _meaningful_char_count(text: str) -> int:
    """Count Unicode letters and digits (includes Hindi and other scripts)."""
    return len(re.findall(r"[\w]", text, re.UNICODE))


def _strip_pdf_wrapper(text: str) -> str:
    return re.sub(r"^#\s+.+\n\nSource:\s+https?://[^\n]+\n\n", "", text.strip(), count=1)


def _baseline_text(raw: RawArtifact) -> str:
    text = raw.content
    if raw.source_type == SourceType.HTML:
        if raw.website == "earkart.in":
            from app.kb.cleaning.earkart_in import _strip_footer, _strip_header_nav

            text, _ = _strip_footer(text)
            text, _ = _strip_header_nav(text)
        elif raw.website == "earkart.com":
            from app.kb.cleaning.earkart_com import (
                _strip_shopify_cart,
                _strip_shopify_footer,
                _strip_shopify_nav_lines,
            )

            text, _ = _strip_shopify_cart(text)
            text, _ = _strip_shopify_nav_lines(text)
            text, _ = _strip_shopify_footer(text)
    elif raw.source_type in (SourceType.PDF, SourceType.SCANNED_PDF):
        text = _strip_pdf_wrapper(text)
    return text


def baseline_text(raw: RawArtifact) -> str:
    """De-boilerplated raw text used for content-loss comparison."""
    return _baseline_text(raw)


def validate_document(raw: RawArtifact, cleaned: CleanedArtifact, canonical: CanonicalDocument) -> ValidationResult:
    warnings: list[str] = []
    errors: list[str] = []

    exclusion = detect_exclusion(raw)
    if exclusion is not None:
        return ValidationResult(
            passed=True,
            status=ProcessingStatus.EXCLUDED,
            warnings=[f"excluded from production KB: {exclusion.value}"],
            excluded=True,
            exclusion_reason=exclusion,
        )

    if not cleaned.content.strip():
        errors.append("cleaned document is empty")

    if not canonical.title.strip():
        warnings.append("missing title")

    if not canonical.source_url:
        errors.append("missing source_url")

    if canonical.source_type not in (SourceType.HTML, SourceType.PDF, SourceType.SCANNED_PDF):
        errors.append("invalid source_type")

    cleaned_meaningful = _meaningful_char_count(cleaned.content)
    baseline_meaningful = _meaningful_char_count(_baseline_text(raw))

    if raw.source_type == SourceType.HTML and cleaned_meaningful < MIN_CLEANED_MEANINGFUL_CHARS:
        errors.append(f"cleaned content too short ({cleaned_meaningful} meaningful chars)")

    skip_retention_check = False
    if raw.source_type in (SourceType.PDF, SourceType.SCANNED_PDF):
        if baseline_meaningful < MIN_SHORT_PDF_BASELINE_CHARS:
            skip_retention_check = True
            if cleaned_meaningful < 30:
                warnings.append(
                    f"sparse PDF extraction ({cleaned_meaningful} meaningful chars) — may be vector/graphic PDF"
                )

    if not skip_retention_check and baseline_meaningful > 0:
        ratio = cleaned_meaningful / baseline_meaningful
        if ratio < MIN_BASELINE_RETENTION_RATIO:
            errors.append(
                f"excessive content loss vs de-boilerplated baseline: retained {ratio:.0%} "
                f"(threshold {MIN_BASELINE_RETENTION_RATIO:.0%})"
            )
        elif ratio < 0.70:
            warnings.append(
                f"high content reduction vs baseline: retained {ratio:.0%} of meaningful characters"
            )

    if canonical.structured_content.pages:
        page_numbers = [page.page_number for page in canonical.structured_content.pages]
        if page_numbers != sorted(page_numbers):
            errors.append("pdf pages out of order")
        if len(set(page_numbers)) != len(page_numbers):
            errors.append("duplicate pdf page numbers")

    boilerplate_markers = [
        "Your cart is empty",
        "### Quick Links",
        "Become Our Partner",
        "[Judge.me]",
    ]
    for marker in boilerplate_markers:
        if marker in cleaned.content:
            warnings.append(f"possible remaining boilerplate: {marker}")

    if errors:
        return ValidationResult(passed=False, status=ProcessingStatus.FAILED, warnings=warnings, errors=errors)
    if warnings:
        return ValidationResult(passed=True, status=ProcessingStatus.VALIDATED, warnings=warnings)
    return ValidationResult(passed=True, status=ProcessingStatus.ACTIVE)
