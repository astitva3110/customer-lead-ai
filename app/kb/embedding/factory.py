"""Embedding provider factory."""

from __future__ import annotations

from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.provider import EmbeddingProvider
from app.kb.embedding.providers.qwen_local import QwenLocalEmbeddingProvider


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "qwen_local":
        return QwenLocalEmbeddingProvider(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
