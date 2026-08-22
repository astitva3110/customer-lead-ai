"""Batch embedding with CUDA OOM recovery and progress logging."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.kb.embedding.device import clear_cuda_cache, is_cuda_oom_error, peak_gpu_memory_mb

logger = logging.getLogger(__name__)


@dataclass
class BatchEmbeddingStats:
    initial_batch_size: int
    final_batch_size: int
    batch_size_history: list[int] = field(default_factory=list)
    oom_retries: int = 0
    peak_gpu_memory_mb: float | None = None
    batches_completed: int = 0


class BatchEmbeddingRunner:
    """Embed texts in batches with OOM halving retry — never skips input texts."""

    def __init__(
        self,
        provider,
        *,
        initial_batch_size: int,
        device_label: str,
        gpu_name: str | None = None,
    ) -> None:
        self.provider = provider
        self.batch_size = max(1, initial_batch_size)
        self.initial_batch_size = self.batch_size
        self.device_label = device_label
        self.gpu_name = gpu_name
        self.stats = BatchEmbeddingStats(
            initial_batch_size=self.initial_batch_size,
            final_batch_size=self.batch_size,
            batch_size_history=[self.batch_size],
        )
        self._started = time.perf_counter()
        self._chunks_done = 0

    def embed_all(self, texts: list[str], *, progress_offset: int = 0, total_count: int | None = None) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        total = total_count or len(texts)
        index = 0
        batch_number = 0
        total_batches = max(1, (len(texts) + self.batch_size - 1) // self.batch_size)

        while index < len(texts):
            batch_number += 1
            batch = texts[index : index + self.batch_size]
            vectors = self.embed_batch(batch)
            all_vectors.extend(vectors)
            self._log_batch_progress(
                batch_number=batch_number,
                total_batches=total_batches,
                chunk_start=progress_offset + index + 1,
                chunk_end=progress_offset + index + len(batch),
                batch_size=len(batch),
                total_count=total,
            )
            index += len(batch)

        self.stats.final_batch_size = self.batch_size
        self.stats.peak_gpu_memory_mb = peak_gpu_memory_mb()
        return all_vectors

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        attempt_batch_size = min(self.batch_size, len(texts))
        while True:
            try:
                vectors = self._embed_sub_batches(texts, attempt_batch_size)
                self._chunks_done += len(texts)
                self.stats.batches_completed += 1
                return vectors
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
                if attempt_batch_size <= 1:
                    raise RuntimeError("CUDA OOM persisted at batch size 1") from exc
                attempt_batch_size = max(1, attempt_batch_size // 2)
                self.batch_size = attempt_batch_size
                self.stats.final_batch_size = attempt_batch_size
                if attempt_batch_size not in self.stats.batch_size_history:
                    self.stats.batch_size_history.append(attempt_batch_size)
                self.stats.oom_retries += 1
                logger.warning(
                    "CUDA OOM; retrying same batch at reduced batch size %s",
                    attempt_batch_size,
                )
                clear_cuda_cache()

    def _embed_sub_batches(self, texts: list[str], sub_batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), sub_batch_size):
            sub = texts[start : start + sub_batch_size]
            vectors.extend(self.provider.embed_batch(sub))
        return vectors

    def _log_batch_progress(
        self,
        *,
        batch_number: int,
        total_batches: int,
        chunk_start: int,
        chunk_end: int,
        batch_size: int,
        total_count: int,
    ) -> None:
        elapsed = time.perf_counter() - self._started
        rate = self._chunks_done / max(elapsed, 1e-9)
        remaining = max(0, total_count - self._chunks_done)
        eta = remaining / rate if rate > 0 else None
        logger.info(
            "Embedding batch %s/%s | Chunks: %s-%s | Batch size: %s | Device: %s | GPU: %s | "
            "Elapsed: %ss | Rate: %s chunks/sec | ETA: %ss",
            batch_number,
            total_batches,
            chunk_start,
            chunk_end,
            batch_size,
            self.device_label,
            self.gpu_name or "n/a",
            round(elapsed, 2),
            round(rate, 3),
            round(eta, 1) if eta is not None else "n/a",
        )
