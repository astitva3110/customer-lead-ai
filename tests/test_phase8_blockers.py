"""Phase 8 regression tests for blocker resolution."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.enums import (
    DocumentType,
    ExtractionMethod,
    ProcessingStatus,
    RetrievalEligibilityStatus,
    SourceType,
)
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.structured_content import DocumentContent, PageContent, ParagraphNode, SectionNode
from app.kb.retrieval.recovery import recover_structure
from app.kb.retrieval.renderers.common import sanitize_retrieval_text
from app.kb.retrieval.renderers.prospectus import maybe_render_prospectus
from app.kb.retrieval.review_decisions import ReviewDecision, decide_review_document
from app.kb.retrieval.service import RetrievalPreparationService
from app.kb.retrieval.storage import RetrievalStore
from app.kb.storage.canonical_store import CanonicalStore

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)
LOGIN_DOC_ID = "452ab392-062f-5283-ac13-9821daa08f1e"
COLLECTIONS_DOC_ID = "7b02cdfc-08c2-56fe-8972-789cfe926082"
VEN_PRODUCT_DOC_ID = "187d654e-7ceb-5ae1-ba2f-65561956c41d"


def _load_canonical(website: str, document_id: str) -> CanonicalDocument:
    path = Path(__file__).resolve().parents[1] / "data" / "canonical" / website / document_id / "current.json"
    return CanonicalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_sanitize_removes_shopify_navigation_links() -> None:
    raw = (
        "Product info\n\n"
        "[Order Now](https://earkart.com/products/bluup)\n\n"
        "![icon](https://earkart.com/cdn/shop/files/icon.png)\n\n"
        "Battery Life: 20 hours"
    )
    cleaned = sanitize_retrieval_text(raw)
    assert "](http" not in cleaned
    assert "cdn/shop" not in cleaned
    assert "Battery Life: 20 hours" in cleaned


def test_pdf_structure_recovery_splits_caps_headings() -> None:
    body = "Earkart Limited is a hearing care company. " * 20
    text = (
        "INTRODUCTION\n"
        f"{body}\n\n"
        "RISK FACTORS\n"
        "Investment involves risk.\n\n"
        "• First bullet item\n"
        "• Second bullet item"
    )
    content = DocumentContent(
        type="document",
        title="Prospectus",
        pages=[PageContent(page_number=1, blocks=[ParagraphNode(text=text)])],
    )
    recovered, stats = recover_structure(content)
    assert stats.recovered_sections >= 2
    assert stats.recovered_headings >= 2
    section_headings = [node.heading for node in recovered.pages[0].blocks if node.type == "section"]
    assert "INTRODUCTION" in section_headings
    assert "RISK FACTORS" in section_headings


def test_prospectus_renderer_strips_pipeline_hashes() -> None:
    content = DocumentContent(
        type="document",
        title="Earkart Prospectus",
        pages=[
            PageContent(
                page_number=1,
                blocks=[ParagraphNode(text="DRAFT PROSPECTUS\nsha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")],
            )
        ],
    )
    rendered = maybe_render_prospectus(
        title="Earkart-Prospectus.pdf",
        canonical_url="https://earkart.in/investor/ipo/Earkart-Prospectus.pdf",
        structured_content=content,
    )
    assert rendered is not None
    assert "sha256:" not in rendered
    assert "DRAFT PROSPECTUS" in rendered


def test_review_decision_unusable_ocr_keeps_review() -> None:
    from app.kb.retrieval.eligibility import assess_retrieval_eligibility

    canonical = _load_canonical("earkart.in", "029e906e-6845-5e7d-92de-b0e698f2ce35")
    base = assess_retrieval_eligibility(canonical)
    record = decide_review_document(canonical, base)
    assert record.decision == ReviewDecision.KEEP_REVIEW


def test_review_decision_good_investor_policy_approves() -> None:
    from app.kb.retrieval.eligibility import assess_retrieval_eligibility

    canonical = _load_canonical("earkart.in", "14c2928c-53aa-5901-b351-8dfaf607835d")
    base = assess_retrieval_eligibility(canonical)
    record = decide_review_document(canonical, base)
    assert record.decision == ReviewDecision.APPROVE


def test_excluded_documents_have_no_retrieval_files(tmp_path: Path) -> None:
    canonical_store = CanonicalStore(Path(__file__).resolve().parents[1] / "data" / "canonical")
    retrieval_store = RetrievalStore(tmp_path / "retrieval")
    decisions_path = Path(__file__).resolve().parents[1] / "reports" / "phase8_review_decisions.json"
    service = RetrievalPreparationService(
        canonical_store,
        retrieval_store,
        review_decisions_path=decisions_path if decisions_path.exists() else None,
    )
    report = service.prepare_all_production()

    login_path = retrieval_store.document_path(KB_DATASET_VERSION, "earkart.com", LOGIN_DOC_ID)
    collections_path = retrieval_store.document_path(KB_DATASET_VERSION, "earkart.com", COLLECTIONS_DOC_ID)
    assert not login_path.exists()
    assert not collections_path.exists()
    assert report.documents_excluded >= 2


def test_product_page_nav_pollution_reduced(tmp_path: Path) -> None:
    canonical = _load_canonical("earkart.com", VEN_PRODUCT_DOC_ID)
    service = RetrievalPreparationService(
        CanonicalStore(tmp_path),
        RetrievalStore(tmp_path / "retrieval"),
        review_decisions_path=None,
    )
    retrieval = service.prepare_from_canonical(canonical)
    assert retrieval is not None
    assert retrieval.retrieval_text.count("](http") < 10
