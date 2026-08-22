"""Adapt Phase 12 structured documents to chunking engine."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from app.config import settings
from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.engine import ChunkingEngine
from app.kb.chunking.models import ChunkRecord
from app.kb.enums import DocumentType, ExtractionMethod, RetrievalEligibilityStatus, SourceType
from app.kb.hashing import content_hash
from app.kb.ingestion.embedding_input import attach_embedding_metadata, build_embedding_input, embedding_input_hash
from app.kb.ingestion.models import DocumentRecord, ExtractionResult, Phase12ChunkRecord
from app.kb.ingestion.structure import structured_to_retrieval_text
from app.kb.models.structured_content import DocumentContent
from app.kb.retrieval.models import RetrievalDocument, RetrievalProvenance


def _source_type_for_extraction(method: ExtractionMethod) -> SourceType:
    if method in {ExtractionMethod.OCR, ExtractionMethod.MIXED}:
        return SourceType.SCANNED_PDF
    return SourceType.PDF


def build_retrieval_document(
    *,
    record: DocumentRecord,
    extraction: ExtractionResult,
    structured: DocumentContent,
) -> RetrievalDocument:
    retrieval_text = structured_to_retrieval_text(structured)
    return RetrievalDocument(
        document_id=record.document_id,
        document_version=record.document_version,
        kb_dataset_version=settings.phase12_kb_dataset_version,
        title=extraction.title,
        canonical_url=f"document://{record.document_id}/v{record.document_version}",
        source_url=f"file://{record.storage_path}",
        website="uploads",
        source_type=_source_type_for_extraction(extraction.extraction_method),
        extraction_method=extraction.extraction_method,
        document_type=record.document_type,
        eligibility_status=RetrievalEligibilityStatus.ELIGIBLE,
        language=record.language,
        content_hash=content_hash(retrieval_text),
        structured_content=structured,
        retrieval_text=retrieval_text,
        provenance=RetrievalProvenance(),
    )


def chunk_document(
    *,
    record: DocumentRecord,
    extraction: ExtractionResult,
    structured: DocumentContent,
    config: ChunkingConfig | None = None,
) -> tuple[list[Phase12ChunkRecord], list[dict]]:
    retrieval = build_retrieval_document(record=record, extraction=extraction, structured=structured)
    engine = ChunkingEngine(config=config)
    records, suppressed = engine.chunk_document_with_suppressed(retrieval)
    phase12_chunks = [
        _to_phase12_chunk(chunk, retrieval=retrieval, file_hash=record.file_hash, extraction=extraction)
        for chunk in records
    ]
    return phase12_chunks, suppressed


def _to_phase12_chunk(
    chunk: ChunkRecord,
    *,
    retrieval: RetrievalDocument,
    file_hash: str,
    extraction: ExtractionResult,
) -> Phase12ChunkRecord:
    base = Phase12ChunkRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_version=chunk.document_version,
        document_type=chunk.document_type,
        chunking_algorithm_version=settings.phase12_chunking_algorithm_version,
        content=chunk.content,
        embedding_input="",
        embedding_input_hash="",
        token_count=chunk.token_count,
        page_number=chunk.page_number,
        section_path=list(chunk.section_path),
        content_type=chunk.split_method.value,
        source_file_hash=file_hash,
        extraction_method=extraction.extraction_method,
        ocr_used=extraction.ocr_used,
        created_at=datetime.now(timezone.utc),
        embedding_version=settings.phase12_embedding_version,
        split_method=chunk.split_method.value,
        overlap_tokens=chunk.overlap_tokens,
    )
    return attach_embedding_metadata(
        base,
        title=retrieval.title,
        embedding_model=settings.embedding_model,
        embedding_model_revision=settings.embedding_model_revision,
        embedding_dimension=settings.embedding_dimension,
        embedding_version=settings.phase12_embedding_version,
    )


def deterministic_chunk_id(
    *,
    kb_dataset_version: str,
    document_id: str,
    document_version: int,
    chunk_index: int,
    content: str,
) -> str:
    return hashlib.sha256(
        f"{kb_dataset_version}|{document_id}|v{document_version}|{chunk_index}|{content}".encode("utf-8")
    ).hexdigest()
