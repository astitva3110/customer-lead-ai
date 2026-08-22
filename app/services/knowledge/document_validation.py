"""Pure document-quality checks run after extraction, before chunking/embed."""

from __future__ import annotations

from app.kb.enums import ExtractionMethod
from app.kb.ingestion.models import DocumentRecord, ExtractionResult

MIN_EXTRACTED_CHARS = 20
GARBAGE_ALPHA_RATIO = 0.15
_PAGED_METHODS = {
    ExtractionMethod.NATIVE,
    ExtractionMethod.OCR,
    ExtractionMethod.MIXED,
    ExtractionMethod.PDF_TEXT,
}


def _combined_text(extraction: ExtractionResult) -> str:
    return "\n".join(unit.text for unit in extraction.units)


def validate_extracted_document(extraction: ExtractionResult, record: DocumentRecord) -> list[str]:
    issues: list[str] = []
    text = _combined_text(extraction).strip()
    if not extraction.units:
        issues.append("empty_extraction")
    if not text:
        issues.append("empty_text")
    elif len(text) < MIN_EXTRACTED_CHARS:
        issues.append("below_min_text_threshold")
    if text:
        alpha_ratio = sum(char.isalpha() for char in text) / max(len(text), 1)
        if alpha_ratio < GARBAGE_ALPHA_RATIO and len(text) > 20:
            issues.append("excessive_garbage")
    if not record.document_id:
        issues.append("missing_document_id")
    if not record.file_hash or not record.filename:
        issues.append("missing_metadata")
    if (
        extraction.extraction_method in _PAGED_METHODS
        and extraction.page_count > 0
        and extraction.units
        and not any(unit.page_number is not None for unit in extraction.units)
    ):
        issues.append("missing_page_provenance")
    if extraction.units and not any(unit.text.strip() for unit in extraction.units):
        issues.append("structure_missing")
    return issues
