import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.cleaning.earkart_com import EarkartComCleaner
from app.kb.enums import ExclusionReason, ProcessingStatus, SourceType
from app.kb.models.raw import RawArtifact
from app.kb.policy.domain_policy import detect_exclusion, is_target_website, website_policy
from app.kb.services.legacy_importer import legacy_record_to_raw
from app.kb.validation.document_validator import _meaningful_char_count, validate_document
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.raw import CleanedArtifact
from app.kb.enums import ExtractionMethod, WebsitePolicy

FIXTURES = Path(__file__).parent / "fixtures" / "cleaning"


def _load_fixture(name: str) -> RawArtifact:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return legacy_record_to_raw(data, scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_meaningful_char_count_includes_devanagari() -> None:
    text = "ओटिटिस मीडिया hearing aid"
    assert _meaningful_char_count(text) > 10


def test_detect_exclusion_cart_page() -> None:
    artifact = _load_fixture("earkart_com_homepage.json")
    artifact = artifact.model_copy(
        update={
            "url": "https://earkart.com/cart",
            "canonical_url": "https://earkart.com/cart",
            "title": "Your Shopping Cart",
        }
    )
    assert detect_exclusion(artifact) == ExclusionReason.CART_CHECKOUT


def test_detect_exclusion_404_page() -> None:
    artifact = _load_fixture("earkart_com_homepage.json")
    artifact = artifact.model_copy(
        update={
            "url": "https://earkart.com/pages/missing",
            "canonical_url": "https://earkart.com/pages/missing",
            "title": "404 Not Found",
            "content": "404\n# Page not found",
        }
    )
    assert detect_exclusion(artifact) == ExclusionReason.NOT_FOUND_404


def test_non_target_website_policy() -> None:
    assert website_policy("firecrawl.dev") == WebsitePolicy.NON_TARGET
    assert website_policy("earkart.in") == WebsitePolicy.TARGET
    assert not is_target_website("crm.earkart.in")


def test_validation_excludes_cart_without_failed_status() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raw = RawArtifact(
        url="https://earkart.com/cart",
        canonical_url="https://earkart.com/cart",
        website="earkart.com",
        title="Cart",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content="cart content",
        scraped_at=now,
        content_hash="sha256:" + "a" * 64,
    )
    cleaned = CleanedArtifact(
        raw_content_hash=raw.content_hash,
        url=raw.url,
        canonical_url=raw.canonical_url,
        website=raw.website,
        title=raw.title,
        source_type=raw.source_type,
        extraction_method=raw.extraction_method,
        content="small",
        cleaned_at=now,
        content_hash="sha256:" + "b" * 64,
    )
    canonical = CanonicalDocument(
        document_id="id",
        website="earkart.com",
        source_url=raw.url,
        canonical_url=raw.canonical_url,
        title=raw.title,
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content_hash=cleaned.content_hash,
        processing_status=ProcessingStatus.CLEANED,
        scraped_at=now,
        created_at=now,
        updated_at=now,
        structured_content={"type": "document", "title": raw.title, "children": []},
        plain_text="small",
    )
    result = validate_document(raw, cleaned, canonical)
    assert result.status == ProcessingStatus.EXCLUDED
    assert result.exclusion_reason == ExclusionReason.CART_CHECKOUT


def test_warranty_policy_passes_with_fixed_baseline(tmp_path) -> None:
    from app.kb.services.cleaning_pipeline import CleaningPipeline
    from app.kb.storage.canonical_store import CanonicalStore
    from app.kb.storage.cleaned_store import CleanedStore
    from app.kb.storage.raw_store import RawStore

    raw = _load_fixture("earkart_com_privacy_policy.json")
    raw_store = RawStore(tmp_path / "raw")
    cleaned_store = CleanedStore(tmp_path / "cleaned")
    canonical_store = CanonicalStore(tmp_path / "canonical")
    raw_store.write(raw)
    _, canonical = CleaningPipeline(cleaned_store, canonical_store).process(raw)
    assert canonical.processing_status in (ProcessingStatus.ACTIVE, ProcessingStatus.VALIDATED)


def test_blog_share_widget_removed() -> None:
    artifact = _load_fixture("earkart_com_product_tiny.json")
    content = artifact.content.replace(
        "# TINY",
        "# TINY\nShare  Share\nLink\nClose share\nCopy link\n",
        1,
    )
    artifact = artifact.model_copy(update={"content": content})
    result = EarkartComCleaner().clean(artifact)
    assert "Copy link" not in result.content
    assert "share_widget" in result.removed_elements


def test_sparse_pdf_skips_retention_check() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raw = RawArtifact(
        url="https://earkart.in/radius/test.pdf",
        canonical_url="https://earkart.in/radius/test.pdf",
        website="earkart.in",
        title="Radius",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        content="# Radius\n\nSource: https://earkart.in/radius/test.pdf\n\nRADIUS 16\n16 Channels",
        scraped_at=now,
        content_hash="sha256:" + "c" * 64,
    )
    cleaned = CleanedArtifact(
        raw_content_hash=raw.content_hash,
        url=raw.url,
        canonical_url=raw.canonical_url,
        website=raw.website,
        title=raw.title,
        source_type=raw.source_type,
        extraction_method=raw.extraction_method,
        content="RADIUS 16\n16 Channels",
        cleaned_at=now,
        content_hash="sha256:" + "d" * 64,
    )
    canonical = CanonicalDocument(
        document_id="id",
        website="earkart.in",
        source_url=raw.url,
        canonical_url=raw.canonical_url,
        title=raw.title,
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        content_hash=cleaned.content_hash,
        processing_status=ProcessingStatus.CLEANED,
        scraped_at=now,
        created_at=now,
        updated_at=now,
        structured_content={"type": "document", "title": raw.title, "children": []},
        plain_text=cleaned.content,
    )
    result = validate_document(raw, cleaned, canonical)
    assert result.status != ProcessingStatus.FAILED


@pytest.mark.parametrize(
    "url",
    [
        "https://earkart.in/investor/notices/FE-Delhi-April-02--2026-Earkart.pdf",
        "https://earkart.in/investor/bp/WhistleBlower-Policy.pdf",
        "https://earkart.in/investor/gm/FE-Delhi-June-25-2026.pdf",
        "https://earkart.in/investor/gm/JS-Delhi-25-June-2026.pdf",
        "https://earkart.in/investor/bp/Code-of-Conduct-Policy.pdf",
        "https://earkart.in/ZB.pdf",
    ],
)
def test_production_excluded_documents(url: str) -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raw = RawArtifact(
        url=url,
        canonical_url=url,
        website="earkart.in",
        title="Excluded PDF",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        content="sample content",
        scraped_at=now,
        content_hash="sha256:" + "e" * 64,
    )
    assert detect_exclusion(raw) == ExclusionReason.NOT_REQUIRED_FOR_PRODUCTION_KB
