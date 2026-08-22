"""Phase 10.6 chunk quality fix regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.duplicates import (
    DuplicateClassification,
    classify_duplicate_pair,
    duplicate_content_key,
    suppress_same_context_duplicates,
)
from app.kb.chunking.engine import ChunkingEngine
from app.kb.chunking.fidelity import check_policy_fidelity, check_product_spec_fidelity
from app.kb.chunking.models import ChunkProvenance, ChunkRecord, SplitMethod
from app.kb.chunking.noise import classify_chunk_noise, should_suppress_noise
from app.kb.chunking.semantic_units import SemanticUnit, build_semantic_units, _merge_heading_with_following
from app.kb.chunking.tokenizer import CharacterEstimateTokenizer, smart_hard_token_split
from app.kb.enums import DocumentType, ExtractionMethod, SourceType
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.models.structured_content import DocumentContent, HeadingNode, ParagraphNode
from app.kb.retrieval.models import RetrievalDocument

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


def _make_chunk(content: str, *, section_path: list[str] | None = None, document_id: str = "doc") -> ChunkRecord:
    return ChunkRecord(
        chunk_id="pending",
        document_id=document_id,
        document_version=1,
        kb_dataset_version=KB_DATASET_VERSION,
        chunk_index=0,
        website="earkart.com",
        document_type=DocumentType.WEBPAGE,
        title="Test",
        section_path=section_path or ["Test"],
        content=content,
        token_count=CharacterEstimateTokenizer(chars_per_token=4.0).count(content),
        split_method=SplitMethod.PARAGRAPH,
        source_url="https://example.com",
        canonical_url="https://example.com",
        page_number=None,
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        provenance=ChunkProvenance(content_hash="sha256:" + "a" * 64),
    )


@pytest.mark.parametrize("tail_words", [4, 10, 19])
def test_orphan_tail_prevention_by_word_count(tail_words: int) -> None:
    tokenizer = CharacterEstimateTokenizer(chars_per_token=4.0)
    config_min = 8
    head = "word " * 500
    tail = "tail " * tail_words
    text = f"{head}{tail}"
    parts = smart_hard_token_split(
        text.strip(),
        max_tokens=512,
        min_meaningful_tokens=config_min,
        tokenizer=tokenizer,
    )
    assert len(parts) >= 1
    assert all(tokenizer.count(part) <= 512 for part in parts)
    if len(parts) > 1:
        assert tokenizer.count(parts[-1]) >= config_min or tokenizer.count(" ".join(parts[-2:])) <= 512


def test_orphan_tail_valid_standalone_short_content_preserved() -> None:
    tokenizer = CharacterEstimateTokenizer(chars_per_token=4.0)
    text = "Battery Size: 13"
    parts = smart_hard_token_split(
        text,
        max_tokens=512,
        min_meaningful_tokens=8,
        tokenizer=tokenizer,
    )
    assert parts == [text]


def test_max_token_constraint_on_smart_split() -> None:
    tokenizer = CharacterEstimateTokenizer(chars_per_token=4.0)
    text = "content " * 600
    parts = smart_hard_token_split(
        text.strip(),
        max_tokens=512,
        min_meaningful_tokens=8,
        tokenizer=tokenizer,
    )
    assert len(parts) > 1
    assert all(tokenizer.count(part) <= 512 for part in parts)


def test_heading_attachment_merge() -> None:
    units = [
        SemanticUnit(content="BATTERY", unit_type="heading", section_path=["Spec", "BATTERY"]),
        SemanticUnit(
            content="Battery Life: 270 Hrs\nBattery Size: 13",
            unit_type="paragraph",
            section_path=["Spec", "BATTERY"],
        ),
    ]
    merged = _merge_heading_with_following(units)
    assert len(merged) == 1
    assert "BATTERY" in merged[0].content
    assert "Battery Life: 270 Hrs" in merged[0].content


def test_parent_section_label_merged_with_first_subsection() -> None:
    section_child = SemanticUnit(
        content="The Customer confirms they are at least 18 years of age.",
        unit_type="paragraph",
        section_path=["Terms", "2.1 Age and Legal Capacity"],
    )
    units = [
        SemanticUnit(
            content="2. Eligibility",
            unit_type="paragraph",
            section_path=["Terms"],
        ),
        SemanticUnit(
            content="",
            unit_type="section",
            section_path=["Terms", "2.1 Age and Legal Capacity"],
            children=[section_child],
        ),
    ]
    merged = _merge_heading_with_following(units)
    assert len(merged) == 1
    assert merged[0].unit_type == "section"
    assert "2. Eligibility" in merged[0].children[0].content
    assert "at least 18 years" in merged[0].children[0].content


def test_heading_attachment_from_structured_content() -> None:
    content = DocumentContent(
        type="document",
        title="Spec",
        children=[
            HeadingNode(text="BATTERY", level=2),
            ParagraphNode(text="Battery Life: 270 Hrs"),
        ],
    )
    units = build_semantic_units(
        structured_content=content,
        document_type=DocumentType.PRODUCT,
        title="Spec",
        canonical_url="https://earkart.in/radius/test.pdf",
    )
    assert any("BATTERY" in unit.content and "Battery Life" in unit.content for unit in units)


def test_valid_small_semantic_chunk_retained() -> None:
    retrieval = _load_retrieval(RADIUS_DOC_ID, "earkart.in")
    chunks = ChunkingEngine().chunk_document(retrieval)
    combined = "\n".join(c.content for c in chunks)
    assert "Battery Size (Zinc Air) 13" in combined


@pytest.mark.parametrize(
    ("content", "expected_category"),
    [
        ("- 55", "standalone_low_value"),
        ("- eet", "standalone_low_value"),
        ("IX\nX\nXI=", "standalone_low_value"),
        ("Open media 1 in modal", "standalone_low_value"),
        ("1 / of 7", "context_noise"),
    ],
)
def test_ocr_debris_detection(content: str, expected_category: str) -> None:
    classification = classify_chunk_noise(
        content,
        split_method=SplitMethod.LIST,
        token_count=CharacterEstimateTokenizer(chars_per_token=4.0).count(content),
    )
    assert classification.category == expected_category
    assert should_suppress_noise(classification)


def test_ocr_debris_suppressed_from_output() -> None:
    content = DocumentContent(
        type="document",
        title="Debris",
        children=[ParagraphNode(text="- 55")],
    )
    retrieval = RetrievalDocument(
        document_id="debris-test",
        document_version=1,
        kb_dataset_version=KB_DATASET_VERSION,
        title="Debris",
        canonical_url="https://earkart.in/test.pdf",
        source_url="https://earkart.in/test.pdf",
        website="earkart.in",
        source_type=SourceType.PDF,
        extraction_method=ExtractionMethod.PDF_TEXT,
        document_type=DocumentType.INVESTOR,
        content_hash="sha256:" + "c" * 64,
        structured_content=content,
        retrieval_text="",
    )
    chunks, suppressed = ChunkingEngine().chunk_document_with_suppressed(retrieval)
    assert not chunks
    assert suppressed
    assert suppressed[0]["suppression_reason"] == "standalone_low_value"


def test_duplicate_context_classification() -> None:
    first = _make_chunk("Same body text here.", section_path=["A"], document_id="d1")
    same_context = _make_chunk("Same body text here.", section_path=["A"], document_id="d1")
    diff_context = _make_chunk("Same body text here.", section_path=["B"], document_id="d1")

    assert classify_duplicate_pair(first, same_context) == DuplicateClassification.SAME_CONTEXT_DUPLICATE
    assert classify_duplicate_pair(first, diff_context) == DuplicateClassification.DIFFERENT_CONTEXT_REPEAT
    assert duplicate_content_key(first) != duplicate_content_key(diff_context)


def test_same_context_duplicate_suppressed() -> None:
    chunks = [
        _make_chunk("Duplicate text.", section_path=["Section"], document_id="d1"),
        _make_chunk("Duplicate text.", section_path=["Section"], document_id="d1"),
        _make_chunk("Duplicate text.", section_path=["Other"], document_id="d1"),
    ]
    kept, suppressed = suppress_same_context_duplicates(chunks)
    assert len(kept) == 2
    assert len(suppressed) == 1
    assert suppressed[0].classification == DuplicateClassification.SAME_CONTEXT_DUPLICATE


def test_product_specification_preservation() -> None:
    retrieval = _load_retrieval(RADIUS_DOC_ID, "earkart.in")
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert check_product_spec_fidelity(chunks).passed


def test_policy_clause_preservation() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert check_policy_fidelity(chunks).passed


def test_pdf_page_provenance() -> None:
    retrieval = _load_retrieval(RADIUS_DOC_ID, "earkart.in")
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert any(chunk.page_number == 1 for chunk in chunks)


def test_prospectus_hierarchy() -> None:
    prospectus_id = "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9"
    retrieval = _load_retrieval(prospectus_id, "earkart.in")
    chunks = ChunkingEngine(ChunkingConfig(max_chunk_tokens=512)).chunk_document(retrieval)
    assert len(chunks) > 100
    assert all(c.token_count <= 512 for c in chunks)


def test_max_token_enforcement() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    chunks = ChunkingEngine().chunk_document(retrieval)
    assert all(c.token_count <= 512 for c in chunks)


def test_zero_unsafe_orphan_tails_on_regression_docs() -> None:
    config = ChunkingConfig()
    for doc_id, website in [
        (RETURNS_DOC_ID, "earkart.com"),
        (RADIUS_DOC_ID, "earkart.in"),
        ("01e76cb2-317f-509a-b8b5-f473f5bd5400", "earkart.in"),
        ("8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9", "earkart.in"),
    ]:
        retrieval = _load_retrieval(doc_id, website)
        chunks = ChunkingEngine(config).chunk_document(retrieval)
        orphans = [
            c
            for c in chunks
            if c.split_method == SplitMethod.HARD_TOKEN_FALLBACK
            and c.token_count < config.min_meaningful_chunk_tokens
            and len(c.content.split()) < 20
        ]
        assert orphans == [], f"orphan tails in {doc_id}: {[c.content[:40] for c in orphans]}"


def test_deterministic_chunk_ids_and_ordering() -> None:
    retrieval = _load_retrieval(RETURNS_DOC_ID, "earkart.com")
    engine = ChunkingEngine()
    first = engine.chunk_document(retrieval)
    second = engine.chunk_document(retrieval)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_index for c in first] == list(range(len(first)))


def test_shopify_ui_not_in_output() -> None:
    content = DocumentContent(
        type="document",
        title="Product",
        children=[ParagraphNode(text="Open media 1 in modal")],
    )
    retrieval = RetrievalDocument(
        document_id="ui-test",
        document_version=1,
        kb_dataset_version=KB_DATASET_VERSION,
        title="Product",
        canonical_url="https://earkart.com/products/test",
        source_url="https://earkart.com/products/test",
        website="earkart.com",
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        document_type=DocumentType.PRODUCT,
        content_hash="sha256:" + "d" * 64,
        structured_content=content,
        retrieval_text="",
    )
    chunks, suppressed = ChunkingEngine().chunk_document_with_suppressed(retrieval)
    assert not any("open media" in c.content.lower() for c in chunks)
    assert suppressed
