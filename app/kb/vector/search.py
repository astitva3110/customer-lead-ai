"""Semantic vector search service."""

from __future__ import annotations

from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.provider import EmbeddingProvider
from app.kb.vector.store import VectorStore


class VectorSearchService:
    def __init__(
        self,
        *,
        config: EmbeddingConfig,
        store: VectorStore,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.provider = provider or create_embedding_provider(config)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        website: str | None = None,
        document_type: str | None = None,
        source_type: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        vectors = self.provider.embed_queries([query])
        if not vectors:
            return []
        return self.store.search(
            vectors[0],
            top_k=top_k,
            embedding_version=self.config.embedding_version,
            website=website,
            document_type=document_type,
            source_type=source_type,
            document_id=document_id,
        )
