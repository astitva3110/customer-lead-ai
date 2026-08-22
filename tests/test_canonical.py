from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.kb.enums import ExtractionMethod, ProcessingStatus, SourceType
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.structured_content import DocumentContent, ParagraphNode


def _sample_document(**overrides) -> CanonicalDocument:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base = {
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "website": "earkart.in",
        "source_url": "https://earkart.in/about-us.html",
        "canonical_url": "https://earkart.in/about-us.html",
        "title": "About Us",
        "source_type": SourceType.HTML,
        "extraction_method": ExtractionMethod.HTML_PARSER,
        "language": "en",
        "version": 1,
        "is_current": True,
        "content_hash": "sha256:" + "a" * 64,
        "processing_status": ProcessingStatus.EXTRACTED,
        "scraped_at": now,
        "created_at": now,
        "updated_at": now,
        "metadata": {},
        "structured_content": DocumentContent(
            type="document",
            title="About Us",
            children=[ParagraphNode(text="Hello")],
        ),
        "plain_text": "Hello",
        "links": [],
    }
    base.update(overrides)
    return CanonicalDocument(**base)


def test_canonical_document_valid() -> None:
    doc = _sample_document()
    assert doc.source_type == SourceType.HTML
    assert doc.extraction_method == ExtractionMethod.HTML_PARSER


@pytest.mark.parametrize("source_type", ["html", "pdf", "scanned_pdf"])
def test_source_type_validation(source_type: str) -> None:
    doc = _sample_document(source_type=source_type)
    assert doc.source_type.value == source_type


@pytest.mark.parametrize("method", ["html_parser", "pdf_text", "ocr"])
def test_extraction_method_validation(method: str) -> None:
    doc = _sample_document(extraction_method=method)
    assert doc.extraction_method.value == method


def test_invalid_content_hash_rejected() -> None:
    with pytest.raises(ValidationError):
        _sample_document(content_hash="md5:deadbeef")


def test_invalid_source_type_rejected() -> None:
    with pytest.raises(ValidationError):
        _sample_document(source_type="docx")
