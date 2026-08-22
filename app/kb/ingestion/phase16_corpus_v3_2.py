"""Phase 16.2 V3.2 corpus — V3.1 plus two catalog summary chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.kb.chunking.v3.embedding_input import attach_v3_embedding_metadata
from app.kb.ingestion.chunking import deterministic_chunk_id
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.phase16_corpus import Phase16Corpus, Phase16CorpusDocument
from app.kb.ingestion.phase16_corpus_v3_1 import SUMMARY_CONTENT_TYPE, load_phase16_v3_1_corpus
from app.kb.ingestion.phase16_v3_2_summaries import (
    build_hearing_aid_catalog_summary_content,
    build_product_catalog_summary_content,
)

V3_2_ALGORITHM_VERSION = "phase16.2.0"
V3_2_KB_DATASET_VERSION = "phase16.2-v1"
PARENT_VERSION = "v3.1"

PRODUCT_CATALOG_SECTION = ["Products", "Product Catalog Summary"]
HEARING_AID_CATALOG_SECTION = ["Hearing Aids", "Hearing Aid Catalog Summary"]
PRODUCT_CATALOG_KNOWLEDGE_KEY = "catalog:products"
HEARING_AID_CATALOG_KNOWLEDGE_KEY = "catalog:hearing_aids"


@dataclass(frozen=True)
class V32IntegrityReport:
    v3_1_total_chunks: int
    v3_1_atomic_chunks: int
    v3_1_summary_chunks: int
    v3_2_total_chunks: int
    v3_2_new_summary_chunks: int
    unchanged_v3_1_chunks: int
    changed_existing_chunks: int
    passed: bool
    details: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_version": PARENT_VERSION,
            "corpus_version": "v3.2",
            "chunking_algorithm_version": V3_2_ALGORITHM_VERSION,
            "kb_dataset_version": V3_2_KB_DATASET_VERSION,
            "v3_1_total_chunks": self.v3_1_total_chunks,
            "v3_1_atomic_chunks": self.v3_1_atomic_chunks,
            "v3_1_summary_chunks": self.v3_1_summary_chunks,
            "v3_2_total_chunks": self.v3_2_total_chunks,
            "v3_2_new_summary_chunks": self.v3_2_new_summary_chunks,
            "unchanged_v3_1_chunks": self.unchanged_v3_1_chunks,
            "changed_existing_chunks": self.changed_existing_chunks,
            "passed": self.passed,
            "details": self.details,
        }


def _chunk_fingerprint(chunk: Phase12ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "embedding_input": chunk.embedding_input,
        "token_count": chunk.token_count,
        "section_path": list(chunk.section_path),
        "content_type": chunk.content_type,
        "chunking_algorithm_version": chunk.chunking_algorithm_version,
    }


def _build_summary_chunk(
    *,
    merged_doc: Phase16CorpusDocument,
    content: str,
    section_path: list[str],
    subsection: str,
    chunk_index: int,
    source_chunk: Phase12ChunkRecord,
) -> Phase12ChunkRecord:
    chunk_id = deterministic_chunk_id(
        kb_dataset_version=V3_2_KB_DATASET_VERSION,
        document_id=merged_doc.record.document_id,
        document_version=merged_doc.record.document_version,
        chunk_index=chunk_index,
        content=content,
    )
    base = Phase12ChunkRecord(
        chunk_id=chunk_id,
        document_id=merged_doc.record.document_id,
        document_version=merged_doc.record.document_version,
        document_type=merged_doc.record.document_type,
        chunking_algorithm_version=V3_2_ALGORITHM_VERSION,
        content=content,
        embedding_input="",
        embedding_input_hash="",
        token_count=max(1, len(content.split())),
        page_number=source_chunk.page_number,
        section_path=section_path,
        parent_section=section_path[0],
        subsection=subsection,
        content_type=SUMMARY_CONTENT_TYPE,
        source_file_hash=merged_doc.record.file_hash,
        extraction_method=source_chunk.extraction_method,
        ocr_used=source_chunk.ocr_used,
        created_at=datetime.now(timezone.utc),
        split_method="v3_summary",
    )
    return attach_v3_embedding_metadata(
        base,
        title=merged_doc.title,
        embedding_model=settings.embedding_model,
        embedding_model_revision=settings.embedding_model_revision,
        embedding_dimension=settings.embedding_dimension,
    )


def verify_v3_2_integrity(
    v3_1_corpus: Phase16Corpus,
    v3_2_corpus: Phase16Corpus,
) -> V32IntegrityReport:
    v3_1_chunks = {chunk.chunk_id: chunk for chunk in v3_1_corpus.all_chunks}
    v3_2_by_id = {chunk.chunk_id: chunk for chunk in v3_2_corpus.all_chunks}
    details: list[str] = []
    changed = 0

    for chunk_id, original in v3_1_chunks.items():
        current = v3_2_by_id.get(chunk_id)
        if current is None:
            details.append(f"missing_v3_1_chunk:{chunk_id}")
            changed += 1
            continue
        if _chunk_fingerprint(original) != _chunk_fingerprint(current):
            details.append(f"modified_v3_1_chunk:{chunk_id}")
            changed += 1

    new_chunk_ids = sorted(set(v3_2_by_id) - set(v3_1_chunks))
    new_summaries = [
        chunk_id
        for chunk_id in new_chunk_ids
        if v3_2_by_id[chunk_id].content_type == SUMMARY_CONTENT_TYPE
    ]

    atomic_v3_1 = sum(1 for chunk in v3_1_corpus.all_chunks if chunk.content_type != SUMMARY_CONTENT_TYPE)
    summary_v3_1 = sum(1 for chunk in v3_1_corpus.all_chunks if chunk.content_type == SUMMARY_CONTENT_TYPE)

    passed = (
        changed == 0
        and len(new_chunk_ids) == 2
        and len(new_summaries) == 2
        and v3_2_corpus.chunk_count == v3_1_corpus.chunk_count + 2
    )
    if len(new_chunk_ids) != 2:
        details.append(f"expected_2_new_chunks_found_{len(new_chunk_ids)}")

    return V32IntegrityReport(
        v3_1_total_chunks=v3_1_corpus.chunk_count,
        v3_1_atomic_chunks=atomic_v3_1,
        v3_1_summary_chunks=summary_v3_1,
        v3_2_total_chunks=v3_2_corpus.chunk_count,
        v3_2_new_summary_chunks=len(new_summaries),
        unchanged_v3_1_chunks=len(v3_1_chunks) - changed,
        changed_existing_chunks=changed,
        passed=passed,
        details=details,
    )


def augment_corpus_with_v3_2_summaries(v3_1_corpus: Phase16Corpus) -> Phase16Corpus:
    merged_doc = next(doc for doc in v3_1_corpus.documents if doc.label == "merged")
    merged_chunks = merged_doc.chunks
    source_chunk = merged_chunks[0]

    product_content = build_product_catalog_summary_content(merged_chunks)
    hearing_aid_content = build_hearing_aid_catalog_summary_content(merged_chunks)

    product_summary = _build_summary_chunk(
        merged_doc=merged_doc,
        content=product_content,
        section_path=PRODUCT_CATALOG_SECTION,
        subsection="Product Catalog Summary",
        chunk_index=len(v3_1_corpus.all_chunks),
        source_chunk=source_chunk,
    )
    hearing_aid_summary = _build_summary_chunk(
        merged_doc=merged_doc,
        content=hearing_aid_content,
        section_path=HEARING_AID_CATALOG_SECTION,
        subsection="Hearing Aid Catalog Summary",
        chunk_index=len(v3_1_corpus.all_chunks) + 1,
        source_chunk=source_chunk,
    )

    documents: list[Phase16CorpusDocument] = []
    for doc in v3_1_corpus.documents:
        if doc.label != "merged":
            documents.append(doc)
            continue
        documents.append(
            Phase16CorpusDocument(
                label=doc.label,
                filename=doc.filename,
                record=doc.record,
                title=doc.title,
                chunks=[*doc.chunks, product_summary, hearing_aid_summary],
            )
        )

    return Phase16Corpus(
        documents=documents,
        suppressed_duplicates=v3_1_corpus.suppressed_duplicates,
        source_pdfs=v3_1_corpus.source_pdfs,
    )


def load_phase16_v3_2_corpus(*, storage_dir=None) -> Phase16Corpus:
    v3_1 = load_phase16_v3_1_corpus(storage_dir=storage_dir)
    v3_2 = augment_corpus_with_v3_2_summaries(v3_1)
    integrity = verify_v3_2_integrity(v3_1, v3_2)
    if not integrity.passed:
        raise RuntimeError(f"V3.2 integrity check failed: {integrity.details}")
    return v3_2


def build_v3_2_integrity_report(*, storage_dir=None) -> V32IntegrityReport:
    v3_1 = load_phase16_v3_1_corpus(storage_dir=storage_dir)
    v3_2 = augment_corpus_with_v3_2_summaries(v3_1)
    return verify_v3_2_integrity(v3_1, v3_2)
