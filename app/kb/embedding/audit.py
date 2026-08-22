"""Embedding audit report models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EmbeddingAuditReport:
    manifest_chunk_count: int = 0
    validated_chunk_count: int = 0
    embedded_count: int = 0
    inserted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    review_skipped: int = 0
    do_not_embed_skipped: int = 0
    vector_dimension: int = 0
    embedding_model: str = ""
    embedding_model_revision: str = ""
    embedding_provider: str = ""
    embedding_version: str = ""
    embedding_input_manifest: str = ""
    chunking_algorithm_version: str = ""
    kb_dataset_version: str = ""
    device: str = ""
    batch_size: int = 0
    normalize: bool = True
    query_instruction_hash: str = ""
    runtime_seconds: float = 0.0
    throughput_chunks_per_second: float = 0.0
    batch_count: int = 0
    database_table: str = ""
    gpu_name: str | None = None
    gpu_total_memory_mb: float | None = None
    peak_gpu_memory_mb: float | None = None
    initial_batch_size: int = 0
    final_batch_size: int = 0
    batch_size_history: list[int] = field(default_factory=list)
    oom_retries: int = 0
    device_info: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def embedding_run_status(self) -> str:
        if self.failed_count > 0:
            return "FAIL"
        if self.validated_chunk_count > 0 and self.embedded_count < self.validated_chunk_count:
            return "FAIL"
        if self.embedded_count == 0 and self.skipped_count == 0:
            return "FAIL"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "embedding_run": self.embedding_run_status,
            "manifest_chunk_count": self.manifest_chunk_count,
            "validated_chunk_count": self.validated_chunk_count,
            "embedded_count": self.embedded_count,
            "inserted_count": self.inserted_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "review_skipped": self.review_skipped,
            "do_not_embed_skipped": self.do_not_embed_skipped,
            "vector_dimension": self.vector_dimension,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_provider": self.embedding_provider,
            "embedding_version": self.embedding_version,
            "embedding_input_manifest": self.embedding_input_manifest,
            "chunking_algorithm_version": self.chunking_algorithm_version,
            "kb_dataset_version": self.kb_dataset_version,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "query_instruction_hash": self.query_instruction_hash,
            "runtime_seconds": self.runtime_seconds,
            "throughput_chunks_per_second": self.throughput_chunks_per_second,
            "batch_count": self.batch_count,
            "database_table": self.database_table,
            "gpu_name": self.gpu_name,
            "gpu_total_memory_mb": self.gpu_total_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "initial_batch_size": self.initial_batch_size,
            "final_batch_size": self.final_batch_size,
            "batch_size_history": self.batch_size_history,
            "oom_retries": self.oom_retries,
            "device_info": self.device_info,
            "errors": self.errors,
        }
