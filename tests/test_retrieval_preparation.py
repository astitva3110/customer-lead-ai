"""Regression tests for Phase 6 retrieval content preparation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.enums import DocumentType, ExtractionMethod, ProcessingStatus, RetrievalEligibilityStatus, SourceType
from app.kb.integrity.contract import downstream_eligible
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.structured_content import (
    DocumentContent,
    ListNode,
    PageContent,
    ParagraphNode,
    SectionNode,
    TableNode,
)
from app.kb.retrieval.classifier import classify_document
from app.kb.retrieval.models import RetrievalDocument
from app.kb.retrieval.recovery import _is_giant_paragraph, recover_structure
from app.kb.retrieval.renderer import render_retrieval_text
from app.kb.retrieval.service import RetrievalPreparationService
from app.kb.retrieval.storage import RetrievalStore
from app.kb.storage.canonical_store import CanonicalStore

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)
RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"


def _load_returns_canonical() -> CanonicalDocument:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "canonical"
        / "earkart.com"
        / RETURNS_DOC_ID
        / "current.json"
    )
    return CanonicalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _make_canonical(**overrides) -> CanonicalDocument:
    base = {
        "document_id": "test-doc",
        "website": "earkart.com",
        "source_url": "https://earkart.com/pages/returns",
        "canonical_url": "https://earkart.com/pages/returns",
        "title": "Return/Replacement Policy",
        "source_type": SourceType.HTML,
        "extraction_method": ExtractionMethod.HTML_PARSER,
        "language": "en",
        "version": 1,
        "is_current": True,
        "content_hash": "sha256:" + "a" * 64,
        "processing_status": ProcessingStatus.ACTIVE,
        "scraped_at": NOW,
        "created_at": NOW,
        "updated_at": NOW,
        "metadata": {"cleaning": {"profile": "earkart.com"}, "validation": {"warnings": []}},
        "structured_content": {"type": "document", "title": "Return/Replacement Policy", "children": []},
        "plain_text": "",
        "links": [],
    }
    base.update(overrides)
    return CanonicalDocument.model_validate(base)


def test_heading_recovery_returns_page() -> None:
    canonical = _load_returns_canonical()
    giant = canonical.structured_content.children[0].children[0]
    assert giant.type == "paragraph"
    assert _is_giant_paragraph(giant.text)

    recovered, stats = recover_structure(canonical.structured_content)
    assert stats.giant_paragraphs_before == 1
    assert stats.giant_paragraphs_after == 0
    assert stats.recovered_sections == 5
    assert stats.recovered_clauses >= 10

    section = recovered.children[0]
    assert section.type == "section"
    assert section.heading == "Return/Replacement Policy"

    child_types = [child.type for child in section.children]
    assert "paragraph" in child_types
    assert "section" in child_types

    nested_headings = [child.heading for child in section.children if child.type == "section"]
    assert "1. Product-Specific Return & Replacement Windows" in nested_headings
    assert "2. General Conditions" in nested_headings
    assert "5. Statutory Rights" in nested_headings


def test_numbered_clause_recovery() -> None:
    text = (
        "**2. General Conditions**\n"
        "2.1 All requests must be supported by a valid invoice/receipt.\n"
        "2.2 Products showing misuse, physical damage, unauthorized repair, or alteration shall not qualify.\n"
        "2.3 Approved returns must be shipped back within five (5) business days from approval."
    )
    content = DocumentContent(
        type="document",
        title="Policy",
        children=[
            SectionNode(
                heading="Policy",
                level=1,
                children=[ParagraphNode(text=text)],
            )
        ],
    )
    recovered, stats = recover_structure(content)
    section = recovered.children[0]
    general = next(child for child in section.children if child.type == "section" and child.heading.startswith("2."))
    clause_texts = [child.text for child in general.children if child.type == "paragraph"]
    assert any(text.startswith("2.1") for text in clause_texts)
    assert any(text.startswith("2.2") for text in clause_texts)
    assert stats.recovered_clauses >= 3


def test_nested_sections_preserved() -> None:
    content = DocumentContent(
        type="document",
        title="Doc",
        children=[
            SectionNode(
                heading="Outer",
                level=1,
                children=[
                    SectionNode(
                        heading="Inner",
                        level=2,
                        children=[ParagraphNode(text="Body")],
                    )
                ],
            )
        ],
    )
    recovered, _ = recover_structure(content)
    outer = recovered.children[0]
    assert outer.type == "section"
    assert outer.children[0].type == "section"
    assert outer.children[0].children[0].text == "Body"


def test_table_preservation() -> None:
    content = DocumentContent(
        type="document",
        title="Product",
        children=[
            SectionNode(
                heading="Specifications",
                level=2,
                children=[
                    TableNode(
                        headers=["Spec", "Value"],
                        rows=[["Battery", "270 Hrs"], ["Channels", "16"]],
                    )
                ],
            )
        ],
    )
    recovered, _ = recover_structure(content)
    table = recovered.children[0].children[0]
    assert table.type == "table"
    assert table.headers == ["Spec", "Value"]
    assert table.rows[0] == ["Battery", "270 Hrs"]


def test_list_preservation() -> None:
    content = DocumentContent(
        type="document",
        title="Doc",
        children=[
            SectionNode(
                heading="Features",
                level=2,
                children=[ListNode(items=["Item A", "Item B"], ordered=False)],
            )
        ],
    )
    recovered, _ = recover_structure(content)
    list_node = recovered.children[0].children[0]
    assert list_node.type == "list"
    assert list_node.items == ["Item A", "Item B"]


def test_pdf_page_provenance() -> None:
    content = DocumentContent(
        type="document",
        title="Spec",
        pages=[
            PageContent(
                page_number=2,
                ocr_confidence=0.91,
                blocks=[ParagraphNode(text="Battery Life: 270 Hrs")],
            )
        ],
    )
    recovered, _ = recover_structure(content)
    assert recovered.pages is not None
    assert recovered.pages[0].page_number == 2
    assert recovered.pages[0].ocr_confidence == 0.91

    retrieval_text = render_retrieval_text(title="Spec", structured_content=recovered)
    assert "Page:\n2" in retrieval_text
    assert "Battery Life: 270 Hrs" in retrieval_text


def test_ocr_provenance_retained(tmp_path: Path) -> None:
    content = DocumentContent(
        type="document",
        title="Radius Spec",
        pages=[
            PageContent(
                page_number=1,
                blocks=[
                    ParagraphNode(
                        text=(
                            "TECHNICAL SPECIFICATIONS\n"
                            "Battery Life: 270 Hrs\n"
                            "Frequency Range: 250 Hz - 8 Khz\n"
                            "Attack Time: 28 ms\n"
                            "Release Time: 891 ms\n"
                            "Battery Size (Zinc Air) 13\n"
                            "Max OSPL90 (dB SPL) 125-134"
                        )
                    )
                ],
            )
        ],
        ocr={"engine": "tesseract", "dpi": 200, "language": "eng", "average_confidence": 0.82},
    )
    canonical = _make_canonical(
        canonical_url="https://earkart.in/radius/RADIUSM16BTE.pdf",
        source_url="https://earkart.in/radius/RADIUSM16BTE.pdf",
        source_type=SourceType.SCANNED_PDF,
        extraction_method=ExtractionMethod.OCR,
        structured_content=content.model_dump(mode="json"),
        plain_text=(
            "TECHNICAL SPECIFICATIONS Battery Life: 270 Hrs Frequency Range: 250 Hz - 8 Khz "
            "Attack Time: 28 ms Release Time: 891 ms Battery Size (Zinc Air) 13 "
            "Max OSPL90 (dB SPL) 125-134 HF Average OSPL 90 Induction Coil Sensitivity"
        ),
    )
    service = RetrievalPreparationService(CanonicalStore(tmp_path), RetrievalStore(tmp_path / "retrieval"))
    retrieval = service.prepare_from_canonical(canonical)
    assert retrieval is not None
    assert retrieval.provenance.ocr is not None
    assert retrieval.provenance.ocr.engine == "tesseract"
    assert retrieval.provenance.ocr.average_confidence == 0.82


def test_metadata_exclusion(tmp_path: Path) -> None:
    canonical = _load_returns_canonical()
    service = RetrievalPreparationService(CanonicalStore(tmp_path), RetrievalStore(tmp_path / "retrieval"))
    retrieval = service.prepare_from_canonical(canonical)
    assert retrieval is not None
    payload = retrieval.model_dump(mode="json")
    assert "metadata" not in payload
    assert "processing_status" not in payload
    assert "scraped_at" not in payload
    assert "cleaning" not in json.dumps(payload)
    assert "validation" not in json.dumps(payload)
    assert "legacy_import" not in json.dumps(payload)


def test_deterministic_output(tmp_path: Path) -> None:
    canonical = _load_returns_canonical()
    service = RetrievalPreparationService(CanonicalStore(tmp_path), RetrievalStore(tmp_path / "retrieval"))
    first = service.prepare_from_canonical(canonical)
    second = service.prepare_from_canonical(canonical)
    assert first is not None and second is not None
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_canonical_data_unchanged(tmp_path: Path) -> None:
    canonical = _load_returns_canonical()
    source_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "canonical"
        / "earkart.com"
        / RETURNS_DOC_ID
        / "current.json"
    )
    before = source_path.read_bytes()
    service = RetrievalPreparationService(CanonicalStore(tmp_path), RetrievalStore(tmp_path / "retrieval"))
    service.prepare_from_canonical(canonical)
    after = source_path.read_bytes()
    assert before == after


def test_document_classification_rules() -> None:
    returns = _load_returns_canonical()
    recovered, _ = recover_structure(returns.structured_content)
    assert (
        classify_document(
            canonical_url=returns.canonical_url,
            title=returns.title,
            source_type=returns.source_type,
            structured_content=recovered,
        )
        == DocumentType.POLICY
    )

    faq_content = DocumentContent(
        type="document",
        title="FAQ's",
        children=[SectionNode(heading="FAQ's", level=2, children=[ParagraphNode(text="Q1. Example?")])],
    )
    assert (
        classify_document(
            canonical_url="https://earkart.com/products/bluup",
            title="Bluup",
            source_type=SourceType.HTML,
            structured_content=faq_content,
        )
        == DocumentType.PRODUCT
    )

    assert (
        classify_document(
            canonical_url="https://earkart.in/investor/gm/Notice.pdf",
            title="Notice",
            source_type=SourceType.PDF,
            structured_content=DocumentContent(type="document", title="Notice", pages=[]),
        )
        == DocumentType.NOTICE
    )


def test_retrieval_text_excludes_pipeline_metadata() -> None:
    canonical = _load_returns_canonical()
    service = RetrievalPreparationService(
        CanonicalStore(Path("/tmp/unused")),
        RetrievalStore(Path("/tmp/unused2")),
    )
    retrieval = service.prepare_from_canonical(canonical)
    assert retrieval is not None
    text = retrieval.retrieval_text
    assert "Return/Replacement Policy" in text
    assert "2. General Conditions" in text
    assert "2.1 All requests" in text
    assert "document_id" not in text
    assert "sha256:" not in text
    assert "Section:" not in text


def test_prepare_all_writes_manifest_and_files(tmp_path: Path) -> None:
    canonical_store = CanonicalStore(Path(__file__).resolve().parents[1] / "data" / "canonical")
    retrieval_store = RetrievalStore(tmp_path / "retrieval")
    service = RetrievalPreparationService(canonical_store, retrieval_store)
    report = service.prepare_all_production()
    assert report.documents_processed == report.documents_eligible + report.documents_review
    assert report.documents_eligible + report.documents_excluded + report.documents_review == 163

    manifest_path = retrieval_store.manifest_path(KB_DATASET_VERSION)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["document_count"] == report.documents_eligible
    assert manifest["kb_dataset_version"] == KB_DATASET_VERSION

    returns_path = retrieval_store.document_path(KB_DATASET_VERSION, "earkart.com", RETURNS_DOC_ID)
    assert returns_path.exists()
    retrieval = RetrievalDocument.model_validate(json.loads(returns_path.read_text(encoding="utf-8")))
    assert retrieval.document_type == DocumentType.POLICY
    assert retrieval.eligibility_status == RetrievalEligibilityStatus.ELIGIBLE
    assert retrieval.structure_recovery.giant_paragraphs_before == 1
    assert retrieval.structure_recovery.giant_paragraphs_after == 0

    login_path = retrieval_store.document_path(
        KB_DATASET_VERSION, "earkart.com", "452ab392-062f-5283-ac13-9821daa08f1e"
    )
    assert not login_path.exists()


def test_downstream_eligible_required() -> None:
    excluded = _make_canonical(processing_status=ProcessingStatus.EXCLUDED)
    service = RetrievalPreparationService(
        CanonicalStore(Path("/tmp/unused")),
        RetrievalStore(Path("/tmp/unused2")),
    )
    assert service.prepare_from_canonical(excluded) is None
    assert not downstream_eligible(excluded)
