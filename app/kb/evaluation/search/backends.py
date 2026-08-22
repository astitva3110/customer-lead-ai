"""Retrieval search backends and embedding session."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.evaluation.models import RankedHit
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.ingestion.indexing import Phase12VectorStore
from app.kb.ingestion.models import Phase12ChunkRecord


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def format_chunk_hit(rank: int, chunk: Phase12ChunkRecord, similarity: float) -> RankedHit:
    return RankedHit(
        rank=rank,
        similarity=round(similarity, 6),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        section_path=list(chunk.section_path),
        token_count=chunk.token_count or estimate_token_count(chunk.content),
        content_type=chunk.content_type,
        text=chunk.content,
        page_number=chunk.page_number,
    )


def format_store_hit(rank: int, item: dict[str, Any]) -> RankedHit:
    return RankedHit(
        rank=rank,
        similarity=float(item.get("similarity") or 0.0),
        chunk_id=item["chunk_id"],
        document_id=item.get("document_id") or "",
        section_path=list(item.get("section_path") or []),
        token_count=estimate_token_count(item.get("content") or ""),
        content_type=item.get("split_method") or item.get("source_type") or "paragraph",
        text=item.get("content") or "",
        page_number=item.get("page_number"),
        document_title=item.get("title") or "",
    )


class EmbeddingSession:
    def __init__(self, *, device: str | None = None) -> None:
        if device:
            import os

            os.environ["EMBEDDING_DEVICE"] = device
        from dataclasses import replace

        config = replace(EmbeddingConfig.from_settings(), device=device or settings.embedding_device)
        self.provider = create_embedding_provider(config)

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        return self.provider.embed_queries(queries)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed_documents(texts)


class RetrievalBackend(ABC):
    @abstractmethod
    def search(self, query: str, *, top_k: int) -> list[RankedHit]:
        raise NotImplementedError


@dataclass
class InMemoryBackend(RetrievalBackend):
    chunks: list[Phase12ChunkRecord]
    vectors: list[list[float]]
    session: EmbeddingSession

    @classmethod
    def from_corpus(
        cls,
        chunks: list[Phase12ChunkRecord],
        *,
        session: EmbeddingSession | None = None,
        device: str | None = None,
    ) -> InMemoryBackend:
        session = session or EmbeddingSession(device=device)
        vectors = session.embed_documents([chunk.embedding_input for chunk in chunks])
        return cls(chunks=chunks, vectors=vectors, session=session)

    def search(self, query: str, *, top_k: int) -> list[RankedHit]:
        query_vector = self.session.embed_queries([query])[0]
        scored = [
            (cosine_similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            format_chunk_hit(index, chunk, similarity)
            for index, (similarity, chunk) in enumerate(scored[:top_k], start=1)
        ]


@dataclass
class PgVectorBackend(RetrievalBackend):
    store: Phase12VectorStore
    session: EmbeddingSession
    embedding_version: str

    @classmethod
    def from_config(cls, config: RetrievalConfig, *, device: str | None = None) -> PgVectorBackend:
        if not config.vector_table:
            raise ValueError("vector_table required for pgvector mode")
        session = EmbeddingSession(device=device)
        store = Phase12VectorStore(table_name=config.vector_table)
        return cls(store=store, session=session, embedding_version=config.embedding_version)

    def search(self, query: str, *, top_k: int) -> list[RankedHit]:
        query_vector = self.session.embed_queries([query])[0]
        raw = self.store.search(
            query_vector,
            top_k=top_k,
            embedding_version=self.embedding_version,
        )
        return [format_store_hit(index, item) for index, item in enumerate(raw, start=1)]


def build_backend(
    config: RetrievalConfig,
    *,
    chunks: list[Phase12ChunkRecord] | None = None,
    device: str | None = None,
) -> RetrievalBackend:
    if config.mode == "in-memory":
        if chunks is None:
            raise ValueError("chunks required for in-memory backend")
        return InMemoryBackend.from_corpus(chunks, device=device)
    return PgVectorBackend.from_config(config, device=device)
