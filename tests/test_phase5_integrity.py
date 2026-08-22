import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.enums import ExclusionReason, ExtractionMethod, ProcessingStatus, SourceType
from app.kb.integrity.checker import check_document_integrity
from app.kb.integrity.consistency import check_content_consistency, plain_text_from_structured
from app.kb.integrity.contract import DownstreamDocument, downstream_eligible
from app.kb.integrity.exclusion import check_exclusion_integrity
from app.kb.integrity.failed_pdfs import assess_failed_pdf
from app.kb.integrity.freeze import CanonicalKbFreezer
from app.kb.integrity.identity import check_identity_integrity
from app.kb.integrity.manifest import build_manifest
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.audit.collect import DocumentBundle
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.raw import RawArtifact
from app.kb.policy.domain_policy import detect_exclusion, is_target_website
from app.kb.services.cleaning_pipeline import CleaningPipeline
from app.kb.services.legacy_importer import legacy_record_to_raw
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore
from app.kb.url_normalizer import make_document_id

FIXTURES = Path(__file__).parent / "fixtures" / "cleaning"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _load_fixture(name: str) -> RawArtifact:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return legacy_record_to_raw(data, scraped_at=NOW)


def _process(tmp_path: Path, name: str) -> CanonicalDocument:
    raw = _load_fixture(name)
    raw_store = RawStore(tmp_path / "raw")
    cleaned_store = CleanedStore(tmp_path / "cleaned")
    canonical_store = CanonicalStore(tmp_path / "canonical")
    raw_store.write(raw)
    _, canonical = CleaningPipeline(cleaned_store, canonical_store).process(raw)
    return canonical


def test_kb_dataset_version_is_set() -> None:
    assert KB_DATASET_VERSION == "2026-08-17-v1"


def test_document_id_is_stable(tmp_path) -> None:
    canonical = _process(tmp_path, "earkart_in_investor_html.json")
    expected = make_document_id(canonical.website, canonical.canonical_url)
    assert canonical.document_id == expected
    result = check_document_integrity(canonical)
    assert result.passed


def test_active_document_passes_integrity(tmp_path) -> None:
    canonical = _process(tmp_path, "earkart_in_terms_html.json")
    assert canonical.processing_status == ProcessingStatus.ACTIVE
    result = check_document_integrity(canonical)
    assert result.passed


def test_non_target_cannot_be_active() -> None:
    doc = CanonicalDocument(
        document_id="id",
        website="firecrawl.dev",
        source_url="https://firecrawl.dev/",
        canonical_url="https://firecrawl.dev/",
        title="Test",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content_hash="sha256:" + "a" * 64,
        processing_status=ProcessingStatus.ACTIVE,
        scraped_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        structured_content={"type": "document", "title": "Test", "children": []},
        plain_text="content here",
    )
    result = check_document_integrity(doc)
    assert not result.passed
    assert any(i.category == "domain" for i in result.issues)


def test_excluded_document_integrity() -> None:
    doc = CanonicalDocument(
        document_id="id",
        website="earkart.com",
        source_url="https://earkart.com/cart",
        canonical_url="https://earkart.com/cart",
        title="Cart",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content_hash="sha256:" + "b" * 64,
        processing_status=ProcessingStatus.EXCLUDED,
        scraped_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        metadata={"excluded": True, "exclusion_reason": ExclusionReason.CART_CHECKOUT.value},
        structured_content={"type": "document", "title": "Cart", "children": []},
        plain_text="cart",
    )
    report = check_exclusion_integrity([doc])
    assert report.passed
    assert len(report.excluded_documents) == 1


def test_excluded_missing_reason_fails() -> None:
    doc = CanonicalDocument(
        document_id="id",
        website="earkart.com",
        source_url="https://earkart.com/cart",
        canonical_url="https://earkart.com/cart",
        title="Cart",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content_hash="sha256:" + "b" * 64,
        processing_status=ProcessingStatus.EXCLUDED,
        scraped_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        metadata={"excluded": True},
        structured_content={"type": "document", "title": "Cart", "children": []},
        plain_text="cart",
    )
    report = check_exclusion_integrity([doc])
    assert not report.passed


def test_duplicate_document_id_detection() -> None:
    base = {
        "website": "earkart.in",
        "source_url": "https://earkart.in/a",
        "canonical_url": "https://earkart.in/a",
        "title": "A",
        "source_type": SourceType.HTML,
        "extraction_method": ExtractionMethod.HTML_PARSER,
        "content_hash": "sha256:" + "a" * 64,
        "processing_status": ProcessingStatus.ACTIVE,
        "scraped_at": NOW,
        "created_at": NOW,
        "updated_at": NOW,
        "structured_content": {"type": "document", "title": "A", "children": []},
        "plain_text": "text",
        "is_current": True,
    }
    d1 = CanonicalDocument(document_id="id-1", **base)
    d2 = CanonicalDocument(document_id="id-1", **{**base, "canonical_url": "https://earkart.in/b"})
    report = check_identity_integrity([d1, d2])
    assert not report.passed
    assert report.duplicate_document_ids


def test_manifest_is_deterministic(tmp_path) -> None:
    c1 = _process(tmp_path, "earkart_in_investor_html.json")
    c2 = _process(tmp_path, "earkart_com_homepage.json")
    m1 = build_manifest([c1, c2])
    m2 = build_manifest([c2, c1])
    assert m1["documents"] == m2["documents"]
    assert m1["kb_dataset_version"] == KB_DATASET_VERSION


def test_downstream_contract_from_canonical(tmp_path) -> None:
    canonical = _process(tmp_path, "earkart_in_terms_html.json")
    downstream = DownstreamDocument.from_canonical(canonical, kb_dataset_version=KB_DATASET_VERSION)
    assert downstream.document_id == canonical.document_id
    assert downstream.plain_text
    assert downstream.kb_dataset_version == KB_DATASET_VERSION
    assert downstream_eligible(canonical)


def test_downstream_not_eligible_for_excluded() -> None:
    doc = CanonicalDocument(
        document_id="id",
        website="earkart.com",
        source_url="https://earkart.com/cart",
        canonical_url="https://earkart.com/cart",
        title="Cart",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        content_hash="sha256:" + "b" * 64,
        processing_status=ProcessingStatus.EXCLUDED,
        scraped_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        metadata={"excluded": True, "exclusion_reason": "cart/checkout page"},
        structured_content={"type": "document", "title": "Cart", "children": []},
        plain_text="cart",
    )
    assert not downstream_eligible(doc)


def test_pdf_page_integrity(tmp_path) -> None:
    canonical = _process(tmp_path, "earkart_in_native_pdf.json")
    result = check_document_integrity(canonical)
    pdf_issues = [i for i in result.issues if i.category == "pdf" and i.severity == "ERROR"]
    assert not pdf_issues


def test_plain_text_consistency_html(tmp_path) -> None:
    canonical = _process(tmp_path, "earkart_in_terms_html.json")
    issues = check_content_consistency(canonical)
    assert not any("empty" in i for i in issues)


def test_unrecoverable_garbled_pdf_assessment() -> None:
    raw = RawArtifact(
        url="https://earkart.in/investor/test.pdf",
        canonical_url="https://earkart.in/investor/test.pdf",
        website="earkart.in",
        title="[PDF] test.pdf",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        content="# test\n\nSource: https://earkart.in/investor/test.pdf\n\n\x01\x02\x03\x04",
        scraped_at=NOW,
        content_hash="sha256:" + "c" * 64,
    )
    from app.kb.models.raw import CleanedArtifact

    cleaned = CleanedArtifact(
        raw_content_hash=raw.content_hash,
        url=raw.url,
        canonical_url=raw.canonical_url,
        website=raw.website,
        title=raw.title,
        source_type=raw.source_type,
        extraction_method=raw.extraction_method,
        content="\x01\x02\x03\x04" * 100,
        cleaned_at=NOW,
        content_hash="sha256:" + "d" * 64,
    )
    canonical = CanonicalDocument(
        document_id=make_document_id(raw.website, raw.canonical_url),
        website=raw.website,
        source_url=raw.url,
        canonical_url=raw.canonical_url,
        title=raw.title,
        source_type=raw.source_type,
        extraction_method=raw.extraction_method,
        content_hash=raw.content_hash,
        processing_status=ProcessingStatus.FAILED,
        scraped_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        structured_content={"type": "document", "title": raw.title, "children": []},
        plain_text=cleaned.content,
    )
    assessment = assess_failed_pdf(DocumentBundle(canonical=canonical, raw=raw, cleaned=cleaned))
    assert assessment.classification == "UNRECOVERABLE"


def test_freeze_excludes_unrecoverable_failed_pdfs(tmp_path) -> None:
    raw_store = RawStore(tmp_path / "raw")
    cleaned_store = CleanedStore(tmp_path / "cleaned")
    canonical_store = CanonicalStore(tmp_path / "canonical")

    raw = RawArtifact(
        url="https://earkart.in/investor/bad.pdf",
        canonical_url="https://earkart.in/investor/bad.pdf",
        website="earkart.in",
        title="[PDF] bad.pdf",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        content="# bad\n\nSource: https://earkart.in/investor/bad.pdf\n\n\x01\x02",
        scraped_at=NOW,
        content_hash="sha256:" + "e" * 64,
    )
    raw_store.write(raw)
    canonical = CanonicalDocument(
        document_id=make_document_id(raw.website, raw.canonical_url),
        website=raw.website,
        source_url=raw.url,
        canonical_url=raw.canonical_url,
        title=raw.title,
        source_type=raw.source_type,
        extraction_method=raw.extraction_method,
        content_hash=raw.content_hash,
        processing_status=ProcessingStatus.FAILED,
        scraped_at=NOW,
        created_at=NOW,
        updated_at=NOW,
        metadata={"raw_content_hash": raw.content_hash},
        structured_content={"type": "document", "title": raw.title, "children": []},
        plain_text="\x01\x02" * 50,
        is_current=True,
    )
    canonical_store.write(canonical)

    freezer = CanonicalKbFreezer(raw_store, cleaned_store, canonical_store, tmp_path / "reports")
    result = freezer.run()
    assert result.excluded_failed_count == 1
    updated = canonical_store.read_current(raw.website, canonical.document_id)
    assert updated.processing_status == ProcessingStatus.EXCLUDED
    assert updated.metadata.get("exclusion_reason") == ExclusionReason.EXTRACTION_FAILED.value


def test_plain_text_from_structured_matches_builder_pattern() -> None:
    structured = {
        "type": "document",
        "title": "T",
        "children": [
            {"type": "section", "heading": "H", "level": 1, "children": [
                {"type": "paragraph", "text": "Body text"}
            ]}
        ],
    }
    from app.kb.models.structured_content import DocumentContent
    text = plain_text_from_structured(DocumentContent.model_validate(structured))
    assert "Body text" in text
    assert "H" in text
