"""Phase 10 document-aware semantic chunking tests."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.engine import ChunkingEngine
from app.kb.chunking.fidelity import check_policy_fidelity, check_product_spec_fidelity
from app.kb.chunking.models import SplitMethod
from app.kb.chunking.semantic_units import build_semantic_units
from app.kb.chunking.service import ChunkingDryRunService
from app.kb.chunking.tokenizer import CharacterEstimateTokenizer, hard_token_split, split_sentences
from app.kb.chunking.validators import validate_chunks
from app.kb.enums import DocumentType, ExtractionMethod, ProcessingStatus, SourceType
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.models.canonical import CanonicalDocument
from app.kb.models.structured_content import DocumentContent, HeadingNode, PageContent, ParagraphNode, SectionNode, TableNode
from app.kb.retrieval.models import RetrievalDocument
from app.kb.retrieval.recovery import recover_structure
from app.kb.retrieval.storage import RetrievalStore

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)
RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"
RADIUS_DOC_ID = "5f2ec8ed-c57d-5511-8d23-df8adf1b548d"


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


def test_semantic_section_grouping_policy() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    units = build_semantic_units(
        structured_content=retrieval.structured_content,
        document_type=retrieval.document_type,
        title=retrieval.title,
        canonical_url=retrieval.canonical_url,
    )
    assert units
    section_headings = []
    for unit in units:
        if unit.unit_type == "section" and unit.section_path:
            section_headings.append(unit.section_path[-1])
        for child in unit.children:
            if child.section_path:
                section_headings.append(child.section_path[-1])
    assert "1. Product-Specific Return & Replacement Windows" in section_headings


def test_context_inheritance_in_chunk_content() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    engine = ChunkingEngine(ChunkingConfig(max_chunk_tokens=512))
    chunks = engine.chunk_document(retrieval)
    clause_chunk = next(c for c in chunks if "1.2.1 Return/Replacement Period" in c.content)
    assert "1. Product-Specific Return & Replacement Windows" in " ".join(clause_chunk.section_path)
    assert "1. Product-Specific Return & Replacement Windows" in clause_chunk.content


def test_token_limit_recursive_split() -> None:
    config = ChunkingConfig(max_chunk_tokens=64)
    engine = ChunkingEngine(config)
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    chunks = engine.chunk_document(retrieval)
    assert len(chunks) > 5
    assert all(c.token_count <= config.max_chunk_tokens for c in chunks)


def test_sentence_fallback() -> None:
    sentences = split_sentences("First sentence here. Second sentence follows. Third sentence ends.")
    assert len(sentences) == 3


def test_hard_token_fallback_last_resort() -> None:
    tokenizer = CharacterEstimateTokenizer(chars_per_token=4)
    parts = hard_token_split("word " * 200, max_tokens=20, tokenizer=tokenizer)
    assert len(parts) > 1


def test_faq_atomicity() -> None:
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
    retrieval = RetrievalDocument(
        document_id="faq-test",
        document_version=1,
        kb_dataset_version=KB_DATASET_VERSION,
        title="FAQ",
        canonical_url="https://earkart.com/products/tiny",
        source_url="https://earkart.com/products/tiny",
        website="earkart.com",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        document_type=DocumentType.FAQ,
        content_hash="sha256:" + "a" * 64,
        structured_content=content,
        retrieval_text="",
    )
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert len(chunks) == 1
    assert "Question:" in chunks[0].content and "Answer:" in chunks[0].content


def test_table_integrity_headers_repeated_on_split() -> None:
    rows = [[f"Spec{i}", f"Value{i}"] for i in range(30)]
    content = DocumentContent(
        type="document",
        title="Specs",
        children=[
            SectionNode(
                heading="Specifications",
                level=2,
                children=[TableNode(headers=["Spec", "Value"], rows=rows)],
            )
        ],
    )
    retrieval = RetrievalDocument(
        document_id="table-test",
        document_version=1,
        kb_dataset_version=KB_DATASET_VERSION,
        title="Specs",
        canonical_url="https://earkart.in/radius/test.pdf",
        source_url="https://earkart.in/radius/test.pdf",
        website="earkart.in",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        document_type=DocumentType.PRODUCT,
        content_hash="sha256:" + "b" * 64,
        structured_content=content,
        retrieval_text="",
    )
    chunks = ChunkingEngine(ChunkingConfig(max_chunk_tokens=80)).chunk_document(retrieval)
    assert len(chunks) > 1
    assert all("Spec | Value" in chunk.content for chunk in chunks)


def test_policy_clause_context_preserved() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    chunks = ChunkingEngine().chunk_document(retrieval)
    fidelity = check_policy_fidelity(chunks)
    assert fidelity.passed


def test_product_specification_preservation() -> None:
    retrieval = _load_retrieval(RADIUS_DOC_ID, "earkart.in")
    chunks = ChunkingEngine().chunk_document(retrieval)
    fidelity = check_product_spec_fidelity(chunks)
    assert fidelity.passed


def test_pdf_page_provenance() -> None:
    retrieval = _load_retrieval(RADIUS_DOC_ID, "earkart.in")
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert any(chunk.page_number == 1 for chunk in chunks)


def test_ocr_provenance_metadata() -> None:
    retrieval = _load_retrieval(RADIUS_DOC_ID, "earkart.in")
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert chunks[0].provenance.ocr_engine == "tesseract"


def test_deterministic_chunk_ids_and_ordering() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    engine = ChunkingEngine()
    first = engine.chunk_document(retrieval)
    second = engine.chunk_document(retrieval)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_index for c in first] == [c.chunk_index for c in second]


def test_duplicate_detection_validator() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    chunks = ChunkingEngine().chunk_document(retrieval)
    result = validate_chunks(chunks, ChunkingConfig())
    duplicate_issues = [i for i in result.issues if i.category == "duplicate_chunk_content"]
    assert not duplicate_issues


def test_prospectus_hierarchy_not_single_chunk(tmp_path: Path) -> None:
    prospectus_id = "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9"
    retrieval = _load_retrieval(prospectus_id, "earkart.in")
    chunks = ChunkingEngine(ChunkingConfig(max_chunk_tokens=512)).chunk_document(retrieval)
    assert len(chunks) > 100
    assert all("sha256:" not in chunk.content.lower() for chunk in chunks)
    assert any(chunk.section_path for chunk in chunks)


def test_zero_overlap_for_normal_semantic_chunks() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    chunks = ChunkingEngine().chunk_document(retrieval)
    semantic = [c for c in chunks if c.split_method != SplitMethod.HARD_TOKEN_FALLBACK]
    assert all(c.overlap_tokens == 0 for c in semantic)


def test_eligible_only_manifest_processing(tmp_path: Path) -> None:
    retrieval_store = RetrievalStore(Path(__file__).resolve().parents[1] / "data" / "retrieval")
    service = ChunkingDryRunService(retrieval_store)
    manifest = service.load_manifest()
    result = service.run()
    assert len(manifest) == 144
    assert result.eligible_documents == 144
    assert result.documents_with_errors == 0
