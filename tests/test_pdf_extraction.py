from __future__ import annotations

import re
from pathlib import Path

import httpx
import pymupdf
import pytest

from app.kb.enums import ExtractionMethod
from app.providers.pdf.pdf_extractor import PdfExtractResult, _configure_tesseract, extract_pdf_bytes
from app.providers.pdf.pdf_layout import extract_page_layout
from app.providers.pdf.pdf_quality import (
    assess_extraction_quality,
    is_poor_quality,
    meaningful_char_count,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"
RADIUS_FIXTURE = FIXTURES / "radius_m16_bte.pdf"

RADIUS_REQUIRED_TERMS = (
    "RADIUS M 16",
    "RA16311",
    "16 Channels and 32 Bands",
    "Fitting Range",
    "Frequency Range",
    "Battery Life",
    "Max OSPL",
    "Battery Size",
    "Attack Time",
    "Release Time",
)


def _has_term(text: str, term: str) -> bool:
    normalized = re.sub(r"[\s.\-_]", "", text.lower())
    term_norm = re.sub(r"[\s.\-_]", "", term.lower())
    return term_norm in normalized


@pytest.fixture(scope="module")
def radius_pdf_bytes() -> bytes:
    assert RADIUS_FIXTURE.exists(), f"missing fixture: {RADIUS_FIXTURE}"
    return RADIUS_FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def tesseract_available() -> bool:
    return _configure_tesseract()


def test_meaningful_char_count() -> None:
    assert meaningful_char_count("RADIUS M 16") >= 9
    assert meaningful_char_count("Battery Size: 13") >= 8


def test_native_radius_extraction_is_sparse(radius_pdf_bytes: bytes) -> None:
    doc = pymupdf.open(stream=radius_pdf_bytes, filetype="pdf")
    try:
        text = doc[0].get_text().strip()
        report = assess_extraction_quality(doc, [text])
        assert report.total_chars < 100
        assert is_poor_quality(report)
    finally:
        doc.close()


def test_quality_detects_vector_graphic_pdf(radius_pdf_bytes: bytes) -> None:
    doc = pymupdf.open(stream=radius_pdf_bytes, filetype="pdf")
    try:
        page = doc[0]
        text = page.get_text().strip()
        report = assess_extraction_quality(doc, [text])
        assert report.pages[0].drawing_count > 100
        assert is_poor_quality(report)
    finally:
        doc.close()


def test_radius_pdf_extraction_contains_specifications(
    radius_pdf_bytes: bytes, tesseract_available: bool
) -> None:
    if not tesseract_available:
        pytest.skip("Tesseract OCR not installed")

    result = extract_pdf_bytes(radius_pdf_bytes, title="RADIUSM16BTE.pdf", ocr_fallback=True)
    assert result.extraction_method == ExtractionMethod.OCR
    assert result.fallback_used is True
    assert result.quality is not None
    assert result.quality.total_chars > 500

    missing = [term for term in RADIUS_REQUIRED_TERMS if not _has_term(result.full_text, term)]
    assert not missing, f"missing terms: {missing}"


def test_radius_extraction_not_empty_without_ocr(radius_pdf_bytes: bytes) -> None:
    result = extract_pdf_bytes(radius_pdf_bytes, title="RADIUSM16BTE.pdf", ocr_fallback=False)
    assert result.pages
    assert "RADIUS" in result.full_text


def test_native_pdf_does_not_force_ocr() -> None:
    """LH71 native PDF should stay on pdf_text extraction."""
    url = "https://earkart.in/LH71-BTE-Kit-Component-CategoryII.pdf"
    with httpx.Client(timeout=60) as client:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        result = extract_pdf_bytes(response.content, title="LH71.pdf", ocr_fallback=True)

    assert result.extraction_method == ExtractionMethod.PDF_TEXT
    assert result.fallback_used is False
    assert result.quality is not None
    assert result.quality.total_chars > 300
    assert "Battery Size" in result.full_text or "Max OSPL" in result.full_text


def test_layout_extraction_preserves_key_value_pairs() -> None:
    url = "https://earkart.in/LH71-BTE-Kit-Component-CategoryII.pdf"
    with httpx.Client(timeout=60) as client:
        response = client.get(url, follow_redirects=True)
        doc = pymupdf.open(stream=response.content, filetype="pdf")
    try:
        text, blocks = extract_page_layout(doc[0])
        assert text
        table_blocks = [block for block in blocks if block.type == "table"]
        assert table_blocks or "Battery Size" in text
    finally:
        doc.close()


def test_extraction_fallback_prefers_richer_result(radius_pdf_bytes: bytes, tesseract_available: bool) -> None:
    if not tesseract_available:
        pytest.skip("Tesseract OCR not installed")

    native = extract_pdf_bytes(radius_pdf_bytes, ocr_fallback=False)
    with_ocr = extract_pdf_bytes(radius_pdf_bytes, ocr_fallback=True)
    assert with_ocr.quality is not None
    assert native.quality is not None
    assert with_ocr.quality.total_chars > native.quality.total_chars * 3


def test_empty_pdf_returns_no_pages() -> None:
    doc = pymupdf.open()
    doc.new_page()
    buffer = doc.write()
    doc.close()
    result = extract_pdf_bytes(buffer, ocr_fallback=False)
    assert isinstance(result, PdfExtractResult)
    assert result.pages == []


def test_scanned_pdf_fixture_quality_threshold() -> None:
    """Existing OCR/scanned PDFs in RAW should score as good quality when re-assessed."""
    import json

    raw_path = Path("data/raw/earkart.in/scanned_pdf")
    if not raw_path.exists():
        pytest.skip("scanned PDF raw data unavailable")

    sample = next(raw_path.glob("sha256_*.json"))
    data = json.loads(sample.read_text(encoding="utf-8"))
    text = re.sub(r"^#\s+.+\n\nSource:\s+https?://[^\n]+\n\n", "", data.get("content", "").strip(), count=1)
    assert meaningful_char_count(text) > 500
