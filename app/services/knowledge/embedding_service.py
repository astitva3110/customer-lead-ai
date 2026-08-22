"""Embedding use case: provider + vector repository, no SDK imports."""

from __future__ import annotations

from dataclasses import dataclass

from app.interfaces.providers.embedding import EmbeddingProvider
from app.interfaces.repositories.vector_repository import VectorRepository
from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord


@dataclass(frozen=True)
class EmbedIndexResult:
    embedded_count: int
    inserted: int
    already_indexed: bool = False
    committed: bool = False


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider, vectors: VectorRepository) -> None:
        self.provider = provider
        self.vectors = vectors

    def embed_and_index(
        self,
        *,
        record: DocumentRecord,
        chunks: list[Phase12ChunkRecord],
        title: str,
        force: bool = False,
    ) -> EmbedIndexResult:
        if not chunks:
            return EmbedIndexResult(embedded_count=0, inserted=0, committed=False)
        self.vectors.ensure_schema()
        if not force and self.vectors.document_vectors_exist(record.document_id, record.document_version):
            return EmbedIndexResult(
                embedded_count=0,
                inserted=0,
                already_indexed=True,
                committed=True,
            )
        embeddings = self.provider.embed_documents([chunk.embedding_input for chunk in chunks])
        inserted = self.vectors.upsert_chunks(
            record=record,
            chunks=chunks,
            embeddings=embeddings,
            title=title,
            force=force,
        )
        return EmbedIndexResult(
            embedded_count=len(chunks),
            inserted=inserted,
            already_indexed=False,
            committed=True,
        )
