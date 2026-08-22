"""V3 embedding input with semantic hierarchy prefix."""

from __future__ import annotations

from app.kb.enums import DocumentType
from app.kb.hashing import sha256_hex
from app.kb.ingestion.models import Phase12ChunkRecord


def build_v3_embedding_input(
    *,
    document_type: DocumentType,
    title: str,
    parent_section: str | None,
    subsection: str | None,
    section_path: list[str],
    content: str,
) -> str:
    lines: list[str] = []
    if title.strip():
        lines.append(f"Document: {title.strip()}")
    lines.append(f"Type: {document_type.value.replace('_', ' ').title()}")
    if parent_section:
        lines.append(f"Parent Section: {parent_section.strip()}")
    if subsection:
        lines.append(f"Subsection: {subsection.strip()}")
    elif section_path:
        lines.append(f"Section: {' > '.join(section_path[-2:])}")
    lines.append("")
    lines.append(content.strip())
    return "\n".join(lines).strip()


def attach_v3_embedding_metadata(
    chunk: Phase12ChunkRecord,
    *,
    title: str,
    embedding_model: str,
    embedding_model_revision: str,
    embedding_dimension: int,
) -> Phase12ChunkRecord:
    embedding_input = build_v3_embedding_input(
        document_type=chunk.document_type,
        title=title,
        parent_section=chunk.parent_section,
        subsection=chunk.subsection,
        section_path=chunk.section_path,
        content=chunk.content,
    )
    return chunk.model_copy(
        update={
            "embedding_input": embedding_input,
            "embedding_input_hash": sha256_hex(embedding_input),
            "embedding_model": embedding_model,
            "embedding_model_revision": embedding_model_revision,
            "embedding_dimension": embedding_dimension,
            "embedding_version": "",
        }
    )
