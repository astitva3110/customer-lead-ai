"""Deterministic embedding pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.embedding.audit import EmbeddingAuditReport
from app.kb.embedding.batch_runner import BatchEmbeddingRunner
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.device import get_device_info, peak_gpu_memory_mb, reset_peak_gpu_memory
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.loader import ChunkLoader
from app.kb.embedding.provider import EmbeddingProvider
from app.kb.embedding.validation import validate_production_chunk
from app.kb.embedding.versioning import VectorIndexIdentity
from app.kb.vector.store import VectorRecordInput, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingPipelineResult:
    audit: EmbeddingAuditReport
    chunks_processed: list[ProductionChunkRecord] = field(default_factory=list)


class EmbeddingPipeline:
    def __init__(
        self,
        *,
        config: EmbeddingConfig,
        loader: ChunkLoader,
        store: VectorStore,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config
        self.loader = loader
        self.store = store
        self.provider = provider or create_embedding_provider(config)
        self.identity = VectorIndexIdentity.from_config(config)

    def run(
        self,
        *,
        chunk_limit: int | None = None,
        force: bool = False,
    ) -> EmbeddingPipelineResult:
        started = time.perf_counter()
        device_info = get_device_info(self.config.device)
        if self.config.device.strip().lower() == "cuda" and not device_info.cuda_available:
            raise RuntimeError("EMBEDDING_DEVICE=cuda requested but CUDA is unavailable")

        reset_peak_gpu_memory()
        logger.info("Embedding device info: %s", device_info.to_dict())

        audit = EmbeddingAuditReport(
            vector_dimension=self.config.dimension,
            embedding_model=self.config.model,
            embedding_model_revision=self.config.model_revision,
            embedding_provider=self.config.provider,
            embedding_version=self.config.embedding_version,
            embedding_input_manifest=self.config.embedding_input_manifest,
            chunking_algorithm_version=self.config.chunking_algorithm_version,
            kb_dataset_version=self.config.kb_dataset_version,
            device=device_info.resolved_device,
            batch_size=self.config.batch_size,
            initial_batch_size=self.config.batch_size,
            final_batch_size=self.config.batch_size,
            normalize=self.config.normalize,
            query_instruction_hash=self.config.query_instruction_hash,
            database_table=self.store.table_name,
            gpu_name=device_info.gpu_name,
            gpu_total_memory_mb=device_info.gpu_total_memory_mb,
            device_info=device_info.to_dict(),
        )

        load_result = self.loader.load_all()
        audit.manifest_chunk_count = load_result.manifest_count
        chunks = load_result.chunks
        if chunk_limit is not None:
            chunks = self.loader.select_deterministic(chunks, chunk_limit)

        validated: list[ProductionChunkRecord] = []
        for chunk in chunks:
            validate_production_chunk(chunk)
            validated.append(chunk)
        audit.validated_chunk_count = len(validated)

        to_embed: list[ProductionChunkRecord] = []
        for chunk in validated:
            if not force and self.store.exists(chunk.chunk_id, self.identity):
                audit.skipped_count += 1
                continue
            to_embed.append(chunk)

        runner = BatchEmbeddingRunner(
            self.provider,
            initial_batch_size=self.config.batch_size,
            device_label=device_info.resolved_device,
            gpu_name=device_info.gpu_name,
        )

        processed: list[ProductionChunkRecord] = []
        index = 0
        total_batches = max(1, (len(to_embed) + runner.batch_size - 1) // max(1, runner.batch_size))
        batch_number = 0

        while index < len(to_embed):
            batch_number += 1
            batch = to_embed[index : index + runner.batch_size]
            texts = [chunk.content for chunk in batch]
            try:
                vectors = runner.embed_batch(texts)
            except Exception as exc:
                audit.failed_count += len(batch)
                audit.errors.append(f"embed_batch:{index}:{exc}")
                raise

            runner._log_batch_progress(
                batch_number=batch_number,
                total_batches=total_batches,
                chunk_start=audit.skipped_count + index + 1,
                chunk_end=audit.skipped_count + index + len(batch),
                batch_size=len(batch),
                total_count=len(validated),
            )

            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding batch size mismatch: expected {len(batch)}, got {len(vectors)}"
                )

            records = [
                VectorRecordInput.from_chunk(
                    chunk=chunk,
                    embedding=vector,
                    identity=self.identity,
                    provider_name=self.provider.provider_name,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            inserted = self.store.upsert_many(records, force=force)
            audit.embedded_count += len(batch)
            audit.inserted_count += inserted
            processed.extend(batch)
            index += len(batch)

        audit.batch_count = batch_number
        audit.final_batch_size = runner.stats.final_batch_size
        audit.batch_size_history = runner.stats.batch_size_history
        audit.oom_retries = runner.stats.oom_retries
        audit.peak_gpu_memory_mb = peak_gpu_memory_mb() or runner.stats.peak_gpu_memory_mb

        elapsed = time.perf_counter() - started
        audit.runtime_seconds = round(elapsed, 3)
        if audit.embedded_count:
            audit.throughput_chunks_per_second = round(audit.embedded_count / max(elapsed, 1e-9), 3)

        logger.info(
            "Embedding complete | total=%s embedded=%s skipped=%s failed=%s time=%ss rate=%s chunks/s "
            "device=%s batch_size=%s oom_retries=%s peak_gpu_mb=%s",
            audit.validated_chunk_count,
            audit.embedded_count,
            audit.skipped_count,
            audit.failed_count,
            audit.runtime_seconds,
            audit.throughput_chunks_per_second,
            audit.device,
            audit.final_batch_size,
            audit.oom_retries,
            audit.peak_gpu_memory_mb,
        )

        return EmbeddingPipelineResult(audit=audit, chunks_processed=processed)
