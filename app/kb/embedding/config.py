"""Embedding configuration."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    model_revision: str
    dimension: int
    batch_size: int
    normalize: bool
    device: str
    query_instruction: str
    kb_dataset_version: str
    chunking_algorithm_version: str
    embedding_input_manifest: str
    embedding_version: str
    database_url: str
    vector_table: str
    vector_smoke_table: str
    vector_benchmark_table: str

    @classmethod
    def from_settings(cls) -> EmbeddingConfig:
        return cls(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            model_revision=settings.embedding_model_revision,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            normalize=settings.embedding_normalize,
            device=settings.embedding_device,
            query_instruction=settings.embedding_query_instruction,
            kb_dataset_version=settings.kb_dataset_version,
            chunking_algorithm_version=settings.chunking_algorithm_version,
            embedding_input_manifest=settings.embedding_input_manifest,
            embedding_version=settings.embedding_version,
            database_url=settings.database_url,
            vector_table=settings.vector_table,
            vector_smoke_table=settings.vector_smoke_table,
            vector_benchmark_table=settings.vector_benchmark_table,
        )

    @property
    def index_identity(self) -> str:
        return (
            f"{self.kb_dataset_version}|{self.chunking_algorithm_version}|"
            f"{self.embedding_input_manifest}|{self.embedding_version}"
        )

    @property
    def query_instruction_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.query_instruction.encode("utf-8")).hexdigest()[:16]
