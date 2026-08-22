import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.cleaning.earkart_com import EarkartComCleaner
from app.kb.cleaning.earkart_in import EarkartInCleaner
from app.kb.cleaning.ocr import OcrCleaner
from app.kb.cleaning.pdf import PdfCleaner
from app.kb.enums import ExtractionMethod, SourceType
from app.kb.models.raw import RawArtifact
from app.kb.services.legacy_importer import legacy_record_to_raw

FIXTURES = Path(__file__).parent / "fixtures" / "cleaning"


def _load_fixture(name: str) -> RawArtifact:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return legacy_record_to_raw(data, scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    ("fixture", "must_contain", "must_not_contain"),
    [
        (
            "earkart_in_investor_html.json",
            ["Material Documents IPO", "CTC KPI"],
            ["Quick Links", "Become Our Partner", "Contact Information"],
        ),
        (
            "earkart_in_terms_html.json",
            ["Terms & Conditions", "General Terms"],
            ["Quick Links"],
        ),
        (
            "earkart_in_blog_html.json",
            ["otitis media", "Hearing"],
            ["Become Our Partner"],
        ),
    ],
)
def test_earkart_in_html_cleaning(fixture, must_contain, must_not_contain) -> None:
    artifact = _load_fixture(fixture)
    result = EarkartInCleaner().clean(artifact)
    lowered = result.content.lower()
    for phrase in must_contain:
        assert phrase.lower() in lowered
    for phrase in must_not_contain:
        assert phrase.lower() not in lowered


def test_earkart_in_native_pdf_preserves_specs() -> None:
    artifact = _load_fixture("earkart_in_native_pdf.json")
    result = EarkartInCleaner().clean(artifact)
    assert "FEATURES" in result.content
    assert "CATEGORY II" in result.content
    assert result.pages is not None
    assert result.pages[0].page_number == 1
    assert "navigation" not in result.removed_elements


def test_earkart_in_ocr_preserves_text_without_spellfix() -> None:
    artifact = _load_fixture("earkart_in_ocr_pdf.json")
    assert artifact.extraction_method.value == "ocr"
    result = OcrCleaner(profile="earkart.in").clean(artifact)
    assert "reterence" in result.content
    assert "NOMINATION AND REMUNERATION" in result.content
    assert "detined" in result.content  # OCR error preserved, not autocorrected to "defined"


@pytest.mark.parametrize(
    ("fixture", "must_contain", "must_not_contain"),
    [
        (
            "earkart_com_homepage.json",
            ["Empowering Hearing Care", "TINY", "Frequently Asked Questions"],
            ["Your cart is empty", "Subtotal", "Check out", "Judge.me"],
        ),
        (
            "earkart_com_product_tiny.json",
            ["TINY", "Channel", "Rechargeable"],
            ["Your cart is empty", "Continue shopping", "[ Home ]"],
        ),
    ],
)
def test_earkart_com_html_cleaning(fixture, must_contain, must_not_contain) -> None:
    artifact = _load_fixture(fixture)
    result = EarkartComCleaner().clean(artifact)
    lowered = result.content.lower()
    for phrase in must_contain:
        assert phrase.lower() in lowered
    for phrase in must_not_contain:
        assert phrase.lower() not in lowered


@pytest.mark.parametrize(
    ("fixture", "must_contain", "must_not_contain"),
    [
        (
            "earkart_com_privacy_policy.json",
            ["Return/Replacement Policy", "Refunds", "consumer protection"],
            ["Your cart is empty", "Judge.me", "## About"],
        ),
    ],
)
def test_earkart_com_policy_cleaning(fixture, must_contain, must_not_contain) -> None:
    artifact = _load_fixture(fixture)
    result = EarkartComCleaner().clean(artifact)
    lowered = result.content.lower()
    for phrase in must_contain:
        assert phrase.lower() in lowered
    for phrase in must_not_contain:
        assert phrase.lower() not in lowered


def test_cleaning_is_deterministic() -> None:
    artifact = _load_fixture("earkart_in_investor_html.json")
    first = EarkartInCleaner().clean(artifact).content
    second = EarkartInCleaner().clean(artifact).content
    assert first == second
    artifact = _load_fixture("earkart_in_investor_html.json")
    first = EarkartInCleaner().clean(artifact).content
    second = EarkartInCleaner().clean(artifact).content
    assert first == second


def test_pdf_wrapper_removed() -> None:
    artifact = _load_fixture("earkart_in_native_pdf.json")
    result = PdfCleaner().clean(artifact)
    assert "Source: https://" not in result.content
    assert "pdf_wrapper" in result.removed_elements or "FEATURES" in result.content
