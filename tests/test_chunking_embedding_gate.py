"""Phase 10.8 embedding gate regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.eligibility import ChunkEligibilityStatus, classify_chunk_eligibility
from app.kb.chunking.engine import ChunkingEngine
from app.kb.chunking.fidelity import (
    RETURNS_DOC_ID,
    RADIUS_M16_DOC_ID,
    check_policy_fidelity,
    check_product_spec_fidelity,
)
from app.kb.chunking.gate import EmbeddingGateService, investigate_p0_fidelity
from app.kb.chunking.models import ProductionChunkRecord
from app.kb.chunking.question_coverage import evaluate_question_coverage
from app.kb.chunking.storage import ChunkStore, chunk_manifest_path
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.models import RetrievalDocument
from app.kb.retrieval.storage import RetrievalStore

EQFY_DOC_ID = "a0fa1902-0871-5f43-9924-69da7d6036c2"
DRAFT_PROSPECTUS_ID = "01e76cb2-317f-509a-b8b5-f473f5bd5400"


def _load_retrieval(document_id: str, website: str) -> RetrievalDocument:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "retrieval"
        / KB_DATASET_VERSION
        / website
        / f"{document_id}.json"
    )
    return RetrievalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def gate_result():
    retrieval_store = RetrievalStore(settings.retrieval_dir)
    chunk_store = ChunkStore(settings.chunks_dir)
    service = EmbeddingGateService(retrieval_store, chunk_store, kb_dataset_version=KB_DATASET_VERSION)
    return service.run(write_production=True)


def test_production_manifest_contains_only_embed_ready(gate_result) -> None:
    for chunk in gate_result.production_chunks:
        assert chunk.eligibility_status == ChunkEligibilityStatus.EMBED_READY.value
    statuses = {item.eligibility_status for item in gate_result.classified}
    assert ChunkEligibilityStatus.EMBED_READY in statuses


def test_review_and_do_not_embed_excluded_from_production(gate_result) -> None:
    production_ids = {c.chunk_id for c in gate_result.production_chunks}
    for item in gate_result.classified:
        if item.eligibility_status != ChunkEligibilityStatus.EMBED_READY:
            assert item.chunk.chunk_id not in production_ids


def test_excluded_chunks_have_audit_records(gate_result) -> None:
    excluded = [c for c in gate_result.classified if c.eligibility_status == ChunkEligibilityStatus.DO_NOT_EMBED]
    assert len(gate_result.excluded_records) >= len(excluded)
    for record in gate_result.excluded_records:
        assert "chunk_id" in record or record.get("source") == "engine_pre_filter"
        assert "reasons" in record or "classification" in record


def test_review_chunks_have_deterministic_category(gate_result) -> None:
    for record in gate_result.review_records:
        assert record.get("review_category")
        assert record.get("reasons")


def test_qc08_return_address_single_chunk(gate_result) -> None:
    returns_chunks = [c for c in gate_result.production_chunks if c.document_id == RETURNS_DOC_ID]
    shipping_chunks = [
        c
        for c in returns_chunks
        if "3.3" in c.content and "Sector 62" in c.content and "Securely package" in c.content
    ]
    assert shipping_chunks, "Expected merged 3.3 clause + shipping address chunk"
    assert len(shipping_chunks) == 1


def test_qc25_drhp_public_issue_in_production_corpus(gate_result) -> None:
    coverage = evaluate_question_coverage(
        [c for c in gate_result.production_chunks],
        review_excluded=[item.chunk for item in gate_result.classified if item.eligibility_status != ChunkEligibilityStatus.EMBED_READY],
    )
    qc25 = next(r for r in coverage["results"] if r["id"] == "QC25")
    assert qc25["knowledge_present_in_production"]
    assert qc25["passed"]


def test_p0_eqfy_false_positive_investigation(gate_result) -> None:
    assert gate_result.p0_findings
    finding = gate_result.p0_findings[0]
    assert finding["document_id"] == EQFY_DOC_ID
    assert finding["classification"].startswith("FALSE_POSITIVE")
    assert not finding["classification"].startswith("REAL_")


def test_product_spec_fidelity_on_production(gate_result) -> None:
    result = check_product_spec_fidelity(gate_result.production_chunks)
    assert result.passed


def test_policy_fidelity_on_production(gate_result) -> None:
    result = check_policy_fidelity(gate_result.production_chunks)
    assert result.passed


def test_question_coverage_35_of_35(gate_result) -> None:
    passed = sum(1 for q in gate_result.question_coverage if q["passed"])
    assert passed == 35


def test_deterministic_chunk_ids(gate_result) -> None:
    retrieval_store = RetrievalStore(settings.retrieval_dir)
    import tempfile
    from pathlib import Path

    chunk_dir = Path(tempfile.mkdtemp())
    service_a = EmbeddingGateService(retrieval_store, ChunkStore(chunk_dir / "a"))
    service_b = EmbeddingGateService(retrieval_store, ChunkStore(chunk_dir / "b"))
    result_a = service_a.run(write_production=False)
    result_b = service_b.run(write_production=False)
    ids_a = [c.chunk_id for c in result_a.production_chunks]
    ids_b = [c.chunk_id for c in result_b.production_chunks]
    assert ids_a == ids_b


def test_production_manifest_invariant_when_gate_passes(gate_result) -> None:
    if not gate_result.gate_passed:
        pytest.skip("Gate did not pass; manifest not created")
    manifest_path = chunk_manifest_path(settings.chunks_dir, KB_DATASET_VERSION)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligibility_status"] == "EMBED_READY"
    for entry in manifest["chunks"]:
        assert entry["eligibility_status"] == "EMBED_READY"


def test_clause_address_merge_semantic_units() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    engine = ChunkingEngine()
    chunks = engine.chunk_document(retrieval)
    merged = [
        c
        for c in chunks
        if "3.3 Securely package" in c.content and "Sector 62" in c.content
    ]
    assert merged


def test_investigate_p0_fidelity_eqfy_direct() -> None:
    retrieval = _load_retrieval(EQFY_DOC_ID, "earkart.in")
    engine = ChunkingEngine()
    chunks = engine.chunk_document(retrieval)
    assert chunks
    finding = investigate_p0_fidelity(chunk=chunks[0], retrieval=retrieval)
    assert finding["classification"].startswith("FALSE_POSITIVE")
