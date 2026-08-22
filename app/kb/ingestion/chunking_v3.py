"""Phase 16 V3 chunking adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from app.config import settings
from app.kb.chunking.v3.embedding_input import attach_v3_embedding_metadata
from app.kb.chunking.v3.engine import ChunkingEngineV3
from app.kb.ingestion.chunking import deterministic_chunk_id
from app.kb.ingestion.models import DocumentRecord, ExtractionResult, Phase12ChunkRecord
from app.kb.ingestion.structure import build_structured_document
from app.kb.models.structured_content import DocumentContent


def chunk_document_v3(
    *,
    record: DocumentRecord,
    extraction: ExtractionResult,
    structured: DocumentContent | None = None,
) -> list[Phase12ChunkRecord]:
    if structured is None:
        structured, _ = build_structured_document(extraction)

    engine = ChunkingEngineV3()
    atoms = engine.build_atomic_units(
        structured_content=structured,
        document_type=record.document_type,
        title=extraction.title,
        canonical_url=f"document://{record.document_id}/v{record.document_version}",
    )

    chunks: list[Phase12ChunkRecord] = []
    for index, atom in enumerate(atoms):
        content = atom.content.strip()
        chunk_id = deterministic_chunk_id(
            kb_dataset_version=settings.phase16_kb_dataset_version,
            document_id=record.document_id,
            document_version=record.document_version,
            chunk_index=index,
            content=content,
        )
        base = Phase12ChunkRecord(
            chunk_id=chunk_id,
            document_id=record.document_id,
            document_version=record.document_version,
            document_type=record.document_type,
            chunking_algorithm_version=settings.phase16_chunking_algorithm_version,
            content=content,
            embedding_input="",
            embedding_input_hash="",
            token_count=engine.tokenizer.count(content),
            page_number=atom.page_number,
            section_path=list(atom.section_path),
            parent_section=atom.parent_section,
            subsection=atom.subsection,
            content_type=atom.unit_type,
            source_file_hash=record.file_hash,
            extraction_method=extraction.extraction_method,
            ocr_used=extraction.ocr_used,
            created_at=datetime.now(timezone.utc),
            split_method="v3_atomic",
        )
        chunks.append(
            attach_v3_embedding_metadata(
                base,
                title=extraction.title,
                embedding_model=settings.embedding_model,
                embedding_model_revision=settings.embedding_model_revision,
                embedding_dimension=settings.embedding_dimension,
            )
        )
    return chunks
