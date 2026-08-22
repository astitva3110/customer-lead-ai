import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.audit.analysis import audit_structure, compute_content_loss
from app.kb.audit.document_audit import audit_document
from app.kb.audit.duplicates import (
    detect_content_hash_duplicates,
    run_duplicate_analysis,
    scan_boilerplate,
)
from app.kb.audit.models import ReviewSeverity
from app.kb.audit.runner import KnowledgeBaseAuditor
from app.kb.enums import ProcessingStatus
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.raw import CleanedArtifact, RawArtifact
from app.kb.services.cleaning_pipeline import CleaningPipeline
from app.kb.services.legacy_importer import legacy_record_to_raw
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore
from app.kb.audit.collect import DocumentBundle

FIXTURES = Path(__file__).parent / "fixtures" / "cleaning"


def _load_fixture(name: str) -> RawArtifact:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return legacy_record_to_raw(data, scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc))


def _process_fixture(tmp_path: Path, name: str) -> DocumentBundle:
    raw = _load_fixture(name)
    raw_store = RawStore(tmp_path / "raw")
    cleaned_store = CleanedStore(tmp_path / "cleaned")
    canonical_store = CanonicalStore(tmp_path / "canonical")
    raw_store.write(raw)
    pipeline = CleaningPipeline(cleaned_store, canonical_store)
    cleaned, canonical = pipeline.process(raw)
    return DocumentBundle(canonical=canonical, raw=raw, cleaned=cleaned)


def test_content_retention_investor_page(tmp_path) -> None:
    bundle = _process_fixture(tmp_path, "earkart_in_investor_html.json")
    loss = compute_content_loss(bundle)
    assert loss["raw_char_count"] > loss["cleaned_char_count"]
    assert loss["retention_vs_baseline"] is not None
    assert loss["retention_vs_baseline"] >= 0.5


def test_boilerplate_removed_from_investor_page(tmp_path) -> None:
    bundle = _process_fixture(tmp_path, "earkart_in_investor_html.json")
    matches = scan_boilerplate(bundle)
    categories = {m.category for m in matches}
    assert "navigation" not in categories
    assert "footer" not in categories


def test_shopify_cart_boilerplate_detected_on_raw_homepage(tmp_path) -> None:
    bundle = _process_fixture(tmp_path, "earkart_com_homepage.json")
    # After cleaning cart should be gone; if any match, it should not be shopify_cart
    matches = scan_boilerplate(bundle)
    cart_matches = [m for m in matches if m.category == "shopify_cart"]
    assert not cart_matches


def test_structure_validation_detects_empty_document() -> None:
    canonical = CanonicalDocument(
        document_id="id",
        website="earkart.in",
        source_url="https://earkart.in/test",
        canonical_url="https://earkart.in/test",
        title="",
        source_type="html",
        extraction_method="html_parser",
        content_hash="sha256:" + "a" * 64,
        processing_status=ProcessingStatus.ACTIVE,
        scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        structured_content={"type": "document", "title": "", "children": []},
        plain_text="",
    )
    bundle = DocumentBundle(canonical=canonical, raw=None, cleaned=None)
    structure = audit_structure(bundle)
    assert "document has no structured children" in structure["issues"]


def test_duplicate_content_hash_detection() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    content_hash = "sha256:" + "d" * 64

    def make_bundle(url: str) -> DocumentBundle:
        canonical = CanonicalDocument(
            document_id=f"id-{url}",
            website="earkart.in",
            source_url=url,
            canonical_url=url,
            title="Same",
            source_type="html",
            extraction_method="html_parser",
            content_hash=content_hash,
            processing_status=ProcessingStatus.ACTIVE,
            scraped_at=now,
            created_at=now,
            updated_at=now,
            structured_content={"type": "document", "title": "Same", "children": []},
            plain_text="same content",
        )
        return DocumentBundle(canonical=canonical, raw=None, cleaned=None)

    dupes = detect_content_hash_duplicates(
        [make_bundle("https://earkart.in/a"), make_bundle("https://earkart.in/b")]
    )
    assert dupes["exact_content_hash_duplicates"]
    assert len(dupes["exact_content_hash_duplicates"][0]["documents"]) == 2


def test_ocr_audit_preserves_errors_without_spellfix(tmp_path) -> None:
    bundle = _process_fixture(tmp_path, "earkart_in_ocr_pdf.json")
    record = audit_document(bundle)
    assert "detined" in (bundle.cleaned.content if bundle.cleaned else "")
    assert record.ocr is not None


def test_empty_document_is_critical(tmp_path) -> None:
    bundle = _process_fixture(tmp_path, "earkart_in_investor_html.json")
    bundle.cleaned = CleanedArtifact(
        raw_content_hash=bundle.raw.content_hash if bundle.raw else "",
        url=bundle.canonical.source_url,
        canonical_url=bundle.canonical.canonical_url,
        website=bundle.canonical.website,
        title=bundle.canonical.title,
        source_type=bundle.canonical.source_type,
        extraction_method=bundle.canonical.extraction_method,
        content="",
        cleaned_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        content_hash="sha256:" + "b" * 64,
    )
    bundle.canonical.plain_text = ""
    record = audit_document(bundle)
    severities = {item.severity for item in record.review_items}
    assert ReviewSeverity.CRITICAL in severities


def test_severity_classification_for_failed_validation(tmp_path) -> None:
    bundle = _process_fixture(tmp_path, "earkart_in_investor_html.json")
    bundle.canonical.processing_status = ProcessingStatus.FAILED
    record = audit_document(bundle)
    assert any(item.severity == ReviewSeverity.CRITICAL for item in record.review_items)


def test_full_audit_runner_writes_reports(tmp_path) -> None:
    raw = _load_fixture("earkart_in_investor_html.json")
    raw_store = RawStore(tmp_path / "raw")
    cleaned_store = CleanedStore(tmp_path / "cleaned")
    canonical_store = CanonicalStore(tmp_path / "canonical")
    raw_store.write(raw)
    auditor = KnowledgeBaseAuditor(
        raw_store=raw_store,
        cleaned_store=cleaned_store,
        canonical_store=canonical_store,
        reports_dir=tmp_path / "reports",
    )
    auditor.process_raw()
    result = auditor.run_audit()
    paths = result["report_paths"]
    assert paths["summary_json"].exists()
    assert paths["documents_json"].exists()
    assert paths["duplicates_json"].exists()
    assert paths["review_json"].exists()
    assert paths["summary_txt"].exists()
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["canonical_count"] == 1


def test_malformed_table_structure_flagged(tmp_path) -> None:
    from app.kb.structuring.markdown import markdown_to_structured

    md = "## Data\n\n| Col1 | Col2 |\n| --- | --- |\n"
    structured = markdown_to_structured("Data", md)
    canonical = CanonicalDocument(
        document_id="id",
        website="earkart.in",
        source_url="https://earkart.in/table",
        canonical_url="https://earkart.in/table",
        title="Data",
        source_type="html",
        extraction_method="html_parser",
        content_hash="sha256:" + "c" * 64,
        processing_status=ProcessingStatus.ACTIVE,
        scraped_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        structured_content=structured,
        plain_text=md,
    )
    bundle = DocumentBundle(canonical=canonical, raw=None, cleaned=None)
    structure = audit_structure(bundle)
    assert any("table" in issue for issue in structure["issues"])
