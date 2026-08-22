"""Phase 14 KB V2 embedding pipeline for Phase 13 validated PDF chunks."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace

from app.config import settings
from app.kb.embedding.audit import EmbeddingAuditReport
from app.kb.embedding.batch_runner import BatchEmbeddingRunner
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.device import get_device_info, peak_gpu_memory_mb, reset_peak_gpu_memory
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.phase13_corpus import Phase13Corpus, Phase13CorpusDocument
from app.kb.ingestion.indexing import Phase12IndexIdentity, Phase12VectorStore
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.quality import validate_chunk
from app.kb.vector.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Phase14EmbeddingResult:
    audit: EmbeddingAuditReport
    embedded_chunks: list[Phase12ChunkRecord] = field(default_factory=list)
    corpus_chunk_count: int = 0
    v1_vector_count_before: int = 0
    v1_vector_count_after: int = 0


def _group_by_document(
    chunks: list[Phase12ChunkRecord],
    corpus: Phase13Corpus,
) -> list[tuple[Phase13CorpusDocument, list[Phase12ChunkRecord]]]:
    by_doc: dict[str, list[Phase12ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.document_id].append(chunk)
    grouped: list[tuple[Phase13CorpusDocument, list[Phase12ChunkRecord]]] = []
    for doc in corpus.documents:
        doc_chunks = sorted(by_doc.get(doc.record.document_id, []), key=lambda c: c.chunk_id)
        if doc_chunks:
            grouped.append((doc, doc_chunks))
    return grouped


class Phase14EmbeddingPipeline:
    """Embed Phase 13 corpus into KB V2 (phase12 vector table) without touching V1."""

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        table_name: str | None = None,
    ) -> None:
        if device:
            import os

            os.environ["EMBEDDING_DEVICE"] = device
        self.batch_size = batch_size or settings.embedding_batch_size
        self.table_name = table_name or settings.phase12_vector_table
        base_config = EmbeddingConfig.from_settings()
        resolved_device = (device or base_config.device).strip().lower()
        self.config = replace(base_config, device=resolved_device)
        self.device = resolved_device
        self.identity = Phase12IndexIdentity.from_settings()

    def run(
        self,
        corpus: Phase13Corpus,
        chunks: list[Phase12ChunkRecord],
        *,
        force: bool = False,
    ) -> Phase14EmbeddingResult:
        if self.device == "cuda":
            device_info = get_device_info("cuda")
            if not device_info.cuda_available:
                raise RuntimeError("CUDA requested but unavailable")
        else:
            device_info = get_device_info(self.device)

        reset_peak_gpu_memory()

        v1_store = VectorStore(
            settings.database_url,
            settings.vector_table,
            settings.embedding_dimension,
        )
        v1_before = v1_store.count(embedding_version=settings.embedding_version)

        store = Phase12VectorStore(table_name=self.table_name)
        store.ensure_schema()

        audit = EmbeddingAuditReport(
            manifest_chunk_count=corpus.chunk_count,
            vector_dimension=settings.embedding_dimension,
            embedding_model=settings.embedding_model,
            embedding_model_revision=settings.embedding_model_revision,
            embedding_provider=settings.embedding_provider,
            embedding_version=settings.phase12_embedding_version,
            embedding_input_manifest=settings.phase12_embedding_input_manifest,
            chunking_algorithm_version=settings.phase12_chunking_algorithm_version,
            kb_dataset_version=settings.phase12_kb_dataset_version,
            device=device_info.resolved_device,
            batch_size=self.batch_size,
            initial_batch_size=self.batch_size,
            final_batch_size=self.batch_size,
            normalize=settings.embedding_normalize,
            query_instruction_hash=EmbeddingConfig.from_settings().query_instruction_hash,
            database_table=self.table_name,
            gpu_name=device_info.gpu_name,
            gpu_total_memory_mb=device_info.gpu_total_memory_mb,
            device_info=device_info.to_dict(),
        )

        validated: list[Phase12ChunkRecord] = []
        for chunk in chunks:
            issues = validate_chunk(chunk)
            if issues:
                audit.failed_count += 1
                audit.errors.append(f"validate:{chunk.chunk_id}:{issues}")
                continue
            validated.append(chunk)
        audit.validated_chunk_count = len(validated)

        to_embed: list[Phase12ChunkRecord] = []
        for chunk in validated:
            if not force and store.exists(chunk.chunk_id):
                audit.skipped_count += 1
                continue
            to_embed.append(chunk)

        provider = create_embedding_provider(self.config)

        runner = BatchEmbeddingRunner(
            provider,
            initial_batch_size=self.batch_size,
            device_label=device_info.resolved_device,
            gpu_name=device_info.gpu_name,
        )

        started = time.perf_counter()
        embedded: list[Phase12ChunkRecord] = []
        index = 0
        total = len(to_embed)

        while index < total:
            batch = to_embed[index : index + runner.batch_size]
            texts = [chunk.embedding_input for chunk in batch]
            try:
                vectors = runner.embed_batch(texts)
            except Exception as exc:
                audit.failed_count += len(batch)
                audit.errors.append(f"embed_batch:{index}:{exc}")
                raise

            if len(vectors) != len(batch):
                raise RuntimeError(f"Batch size mismatch: expected {len(batch)}, got {len(vectors)}")

            for chunk, vector in zip(batch, vectors, strict=True):
                if len(vector) != settings.embedding_dimension:
                    raise RuntimeError(
                        f"Vector dimension mismatch for {chunk.chunk_id}: "
                        f"{len(vector)} != {settings.embedding_dimension}"
                    )

            by_doc_id: dict[str, list[tuple[Phase12ChunkRecord, list[float]]]] = defaultdict(list)
            for chunk, vector in zip(batch, vectors, strict=True):
                by_doc_id[chunk.document_id].append((chunk, vector))

            for doc in corpus.documents:
                pairs = by_doc_id.get(doc.record.document_id, [])
                if not pairs:
                    continue
                pairs.sort(key=lambda item: item[0].chunk_id)
                doc_chunks = [item[0] for item in pairs]
                doc_vectors = [item[1] for item in pairs]
                inserted = store.upsert_chunks(
                    record=doc.record,
                    chunks=doc_chunks,
                    embeddings=doc_vectors,
                    title=doc.title,
                    force=force,
                )
                audit.inserted_count += inserted

            audit.embedded_count += len(batch)
            embedded.extend(batch)
            index += len(batch)

        audit.batch_count = runner.stats.batches_completed
        audit.final_batch_size = runner.stats.final_batch_size
        audit.batch_size_history = runner.stats.batch_size_history
        audit.oom_retries = runner.stats.oom_retries
        audit.peak_gpu_memory_mb = peak_gpu_memory_mb() or runner.stats.peak_gpu_memory_mb
        audit.runtime_seconds = round(time.perf_counter() - started, 3)
        if audit.embedded_count:
            audit.throughput_chunks_per_second = round(
                audit.embedded_count / max(audit.runtime_seconds, 1e-9), 3
            )

        v1_after = v1_store.count(embedding_version=settings.embedding_version)

        logger.info(
            "Phase14 embed complete | validated=%s embedded=%s skipped=%s failed=%s table=%s v1=%s",
            audit.validated_chunk_count,
            audit.embedded_count,
            audit.skipped_count,
            audit.failed_count,
            self.table_name,
            v1_after,
        )

        return Phase14EmbeddingResult(
            audit=audit,
            embedded_chunks=embedded,
            corpus_chunk_count=corpus.chunk_count,
            v1_vector_count_before=v1_before,
            v1_vector_count_after=v1_after,
        )
