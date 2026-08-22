"""Deterministic embedding input construction."""

from __future__ import annotations

from app.kb.enums import DocumentType
from app.kb.hashing import sha256_hex
from app.kb.ingestion.models import Phase12ChunkRecord


def build_embedding_input(
    *,
    document_type: DocumentType,
    title: str,
    section_path: list[str],
    content: str,
    language: str = "en",
) -> str:
    """Build deterministic embedding input with bounded context inheritance."""
    lines: list[str] = []
    doc_label = document_type.value.replace("_", " ").title()
    if title.strip():
        lines.append(f"Document: {title.strip()}")
    lines.append(f"Type: {doc_label}")
    if section_path:
        bounded = section_path[-3:]
        lines.append(f"Section: {' > '.join(bounded)}")
    lines.append("")
    lines.append(content.strip())
    return "\n".join(lines).strip()


def embedding_input_hash(embedding_input: str) -> str:
    return sha256_hex(embedding_input)


def attach_embedding_metadata(
    chunk: Phase12ChunkRecord,
    *,
    title: str,
    embedding_model: str,
    embedding_model_revision: str,
    embedding_dimension: int,
    embedding_version: str,
) -> Phase12ChunkRecord:
    embedding_input = build_embedding_input(
        document_type=chunk.document_type,
        title=title,
        section_path=chunk.section_path,
        content=chunk.content,
    )
    return chunk.model_copy(
        update={
            "embedding_input": embedding_input,
            "embedding_input_hash": embedding_input_hash(embedding_input),
            "embedding_model": embedding_model,
            "embedding_model_revision": embedding_model_revision,
            "embedding_dimension": embedding_dimension,
            "embedding_version": embedding_version,
        }
    )
