"""Tests for retrieval eligibility gate and type-specific rendering."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.enums import (
    DocumentType,
    ExtractionMethod,
    ProcessingStatus,
    RetrievalEligibilityStatus,
    RetrievalExclusionReason,
    RetrievalReviewReason,
    SourceType,
)
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.structured_content import DocumentContent, HeadingNode, ParagraphNode, SectionNode
from app.kb.retrieval.eligibility import assess_retrieval_eligibility
from app.kb.retrieval.renderer import render_retrieval_text
from app.kb.retrieval.service import RetrievalPreparationService
from app.kb.retrieval.storage import RetrievalStore
from app.kb.storage.canonical_store import CanonicalStore

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)
LOGIN_DOC_ID = "452ab392-062f-5283-ac13-9821daa08f1e"
PM_CARES_DOC_ID = "2e4da1d8-2e2c-503e-b6c8-a2f02aaa3313"
RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"


def _load_canonical(website: str, document_id: str) -> CanonicalDocument:
    path = Path(__file__).resolve().parents[1] / "data" / "canonical" / website / document_id / "current.json"
    return CanonicalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_login_page_retrieval_excluded() -> None:
    canonical = _load_canonical("earkart.com", LOGIN_DOC_ID)
    decision = assess_retrieval_eligibility(canonical)
    assert decision.status == RetrievalEligibilityStatus.EXCLUDED
    assert decision.exclusion_reason == RetrievalExclusionReason.ACCOUNT_UI

    service = RetrievalPreparationService(
        CanonicalStore(Path("/tmp/unused")),
        RetrievalStore(Path("/tmp/unused2")),
    )
    assert service.prepare_from_canonical(canonical) is None


def test_pm_cares_retrieval_review_not_excluded() -> None:
    canonical = _load_canonical("earkart.in", PM_CARES_DOC_ID)
    decision = assess_retrieval_eligibility(canonical)
    assert decision.status == RetrievalEligibilityStatus.REVIEW
    assert decision.review_reason == RetrievalReviewReason.OCR_QUALITY_CONCERN
    assert decision.exclusion_reason is None

    service = RetrievalPreparationService(
        CanonicalStore(Path(__file__).resolve().parents[1] / "data" / "canonical"),
        RetrievalStore(Path("/tmp/unused3")),
        review_decisions_path=None,
    )
    retrieval = service.prepare_from_canonical(canonical)
    assert retrieval is not None
    assert retrieval.eligibility_status == RetrievalEligibilityStatus.REVIEW
    assert "PM CARES" in retrieval.retrieval_text or "Receipt" in retrieval.retrieval_text


def test_policy_renderer_uses_compact_format() -> None:
    canonical = _load_canonical("earkart.com", RETURNS_DOC_ID)
    recovered = RetrievalPreparationService(
        CanonicalStore(Path("/tmp/unused4")),
        RetrievalStore(Path("/tmp/unused5")),
    ).prepare_from_canonical(canonical)
    assert recovered is not None
    text = recovered.retrieval_text
    assert "Document:" not in text
    assert "Section:" not in text
    assert "Return/Replacement Policy" in text
    assert "2. General Conditions" in text
    assert "2.1 All requests" in text


def test_faq_renderer_question_answer_format() -> None:
    content = DocumentContent(
        type="document",
        title="FAQ",
        children=[
            SectionNode(
                heading="FAQ's",
                level=2,
                children=[
                    HeadingNode(text="How long does battery last?", level=3),
                    ParagraphNode(text="Up to 26 hours on a full charge."),
                ],
            )
        ],
    )
    text = render_retrieval_text(
        title="FAQ",
        canonical_url="https://earkart.com/products/tiny",
        document_type=DocumentType.FAQ,
        source_type=SourceType.HTML,
        structured_content=content,
    )
    assert "Question:" in text
    assert "Answer:" in text
    assert "How long does battery last?" in text


def test_product_renderer_format() -> None:
    content = DocumentContent(
        type="document",
        title="Tiny",
        children=[
            SectionNode(
                heading="Specifications",
                level=2,
                children=[ParagraphNode(text="Battery Life: 20 hours")],
            )
        ],
    )
    text = render_retrieval_text(
        title="TINY",
        canonical_url="https://earkart.com/products/tiny",
        document_type=DocumentType.PRODUCT,
        source_type=SourceType.HTML,
        structured_content=content,
    )
    assert text.startswith("Product:")
    assert "Specifications" in text
    assert "Battery Life: 20 hours" in text


def test_eligibility_manifest_written(tmp_path: Path) -> None:
    canonical_store = CanonicalStore(Path(__file__).resolve().parents[1] / "data" / "canonical")
    retrieval_store = RetrievalStore(tmp_path / "retrieval")
    service = RetrievalPreparationService(canonical_store, retrieval_store)
    report = service.prepare_all_production()

    eligibility_path = retrieval_store.eligibility_manifest_path(service.kb_dataset_version)
    assert eligibility_path.exists()
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    assert eligibility["document_count"] == 163
    assert eligibility["excluded_count"] == report.documents_excluded
    assert eligibility["review_count"] == report.documents_review
    assert eligibility["eligible_count"] == report.documents_eligible

    login_entry = next(
        item for item in eligibility["documents"] if item["document_id"] == LOGIN_DOC_ID
    )
    assert login_entry["eligibility_status"] == "excluded"

    pm_entry = next(item for item in eligibility["documents"] if item["document_id"] == PM_CARES_DOC_ID)
    assert pm_entry["eligibility_status"] in {"review", "eligible"}

    manifest = json.loads(retrieval_store.manifest_path(service.kb_dataset_version).read_text(encoding="utf-8"))
    assert manifest["document_count"] == report.documents_eligible
    assert all(item["eligibility_status"] == "eligible" for item in manifest["documents"])
    assert not any(item["document_id"] == LOGIN_DOC_ID for item in manifest["documents"])
