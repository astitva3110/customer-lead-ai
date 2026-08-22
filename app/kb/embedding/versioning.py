"""Embedding version identity helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.kb.embedding.config import EmbeddingConfig


@dataclass(frozen=True)
class VectorIndexIdentity:
    kb_dataset_version: str
    chunking_algorithm_version: str
    embedding_input_manifest: str
    embedding_version: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> VectorIndexIdentity:
        return cls(
            kb_dataset_version=config.kb_dataset_version,
            chunking_algorithm_version=config.chunking_algorithm_version,
            embedding_input_manifest=config.embedding_input_manifest,
            embedding_version=config.embedding_version,
            embedding_provider=config.provider,
            embedding_model=config.model,
            embedding_model_revision=config.model_revision,
            embedding_dimension=config.dimension,
        )

    @property
    def unique_key(self) -> str:
        return (
            f"{self.kb_dataset_version}:{self.chunking_algorithm_version}:"
            f"{self.embedding_input_manifest}:{self.embedding_model}:"
            f"{self.embedding_model_revision}:{self.embedding_version}"
        )
