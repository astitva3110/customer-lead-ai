"""Production chunk validation for embedding pipeline."""

from __future__ import annotations

from app.kb.chunking.models import ProductionChunkRecord

REQUIRED_FIELDS = (
    "chunk_id",
    "document_id",
    "document_version",
    "kb_dataset_version",
    "chunk_index",
    "content",
    "token_count",
    "section_path",
    "title",
    "website",
    "document_type",
    "source_url",
    "canonical_url",
    "source_type",
    "extraction_method",
    "split_method",
    "provenance",
    "eligibility_status",
)


class ChunkValidationError(ValueError):
    pass


def validate_production_chunk(chunk: ProductionChunkRecord) -> None:
    data = chunk.model_dump()
    for field in REQUIRED_FIELDS:
        if field not in data or data[field] is None:
            raise ChunkValidationError(f"Missing required field '{field}' on chunk_id={chunk.chunk_id}")

    if chunk.eligibility_status != "EMBED_READY":
        raise ChunkValidationError(
            f"Chunk {chunk.chunk_id} has eligibility_status={chunk.eligibility_status!r}, expected EMBED_READY"
        )
    if not chunk.content.strip():
        raise ChunkValidationError(f"Chunk {chunk.chunk_id} has empty content")
    if not chunk.chunk_id.strip():
        raise ChunkValidationError("Chunk has empty chunk_id")
    if not chunk.document_id.strip():
        raise ChunkValidationError(f"Chunk {chunk.chunk_id} has empty document_id")
    if not chunk.source_url.strip():
        raise ChunkValidationError(f"Chunk {chunk.chunk_id} has empty source_url")
    if not chunk.canonical_url.strip():
        raise ChunkValidationError(f"Chunk {chunk.chunk_id} has empty canonical_url")


def reject_non_embed_ready_status(status: str, *, chunk_id: str = "unknown") -> None:
    if status == "REVIEW":
        raise ChunkValidationError(f"REVIEW chunk rejected: {chunk_id}")
    if status == "DO_NOT_EMBED":
        raise ChunkValidationError(f"DO_NOT_EMBED chunk rejected: {chunk_id}")
    if status != "EMBED_READY":
        raise ChunkValidationError(f"Invalid eligibility status {status!r} for chunk {chunk_id}")
