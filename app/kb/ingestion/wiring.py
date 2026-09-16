"""Default wiring for IngestionService — infrastructure stays out of core."""

from __future__ import annotations

from collections.abc import Callable

from app.config import settings
from app.services.knowledge.embedding_service import EmbeddingService
from app.services.knowledge.ingestion_service import IngestionService
from app.kb.chunking.config import ChunkingConfig
from app.kb.ingestion.adapters import DefaultExtractorFactory, Phase12DocumentChunker, StructuredContentCleaner
from app.kb.retrieval.runtime_config import build_production_vector_store, load_runtime_retrieval_config
from app.kb.ingestion.quality import ChunkQualityGate
from app.kb.ingestion.storage import DocumentStorage

def default_embedding_service() -> EmbeddingService:
    from app.kb.embedding.config import EmbeddingConfig
    from app.kb.embedding.factory import create_embedding_provider

    retrieval = load_runtime_retrieval_config()
    store = build_production_vector_store(retrieval)
    store.ensure_schema()
    store.ensure_content_fts_index()
    return EmbeddingService(create_embedding_provider(EmbeddingConfig.from_settings()), store)


def build_ingestion_service(
    *,
    storage: DocumentStorage | None = None,
    quality_gate: ChunkQualityGate | None = None,
    chunking_config: ChunkingConfig | None = None,
    embedding_service: EmbeddingService | None = None,
    embedding_factory: Callable[[], EmbeddingService] | None = None,
) -> IngestionService:
    documents = storage or DocumentStorage()
    config = chunking_config or ChunkingConfig(
        max_chunk_tokens=settings.chunk_max_tokens,
        emergency_overlap_tokens=settings.chunk_emergency_overlap_tokens,
        chars_per_token=settings.chunk_chars_per_token,
    )
    factory = embedding_factory
    if embedding_service is None and factory is None:
        factory = default_embedding_service
    return IngestionService(
        documents=documents,
        extractor_factory=DefaultExtractorFactory(),
        cleaner=StructuredContentCleaner(),
        chunker=Phase12DocumentChunker(config),
        quality_gate=quality_gate or ChunkQualityGate(),
        embedding_service=embedding_service,
        embedding_factory=factory,
    )
