"""GPU batch-size benchmark for Phase 11 embedding."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from app.kb.embedding.batch_runner import BatchEmbeddingRunner
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.device import get_device_info, reset_peak_gpu_memory
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.loader import ChunkLoader
from app.kb.vector.store import VectorStore

BENCHMARK_CHUNK_COUNT = 100
DEFAULT_BENCHMARK_BATCH_SIZES = (4, 8, 16)


@dataclass
class BenchmarkResult:
    batch_size: int
    success: bool
    total_seconds: float
    chunks_per_second: float
    average_batch_seconds: float
    peak_gpu_memory_mb: float | None
    oom_occurred: bool
    oom_retries: int
    error: str | None = None


@dataclass
class BenchmarkReport:
    device_info: dict
    chunk_count: int
    model: str
    model_revision: str
    dimension: int
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "device_info": self.device_info,
            "chunk_count": self.chunk_count,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "results": [
                {
                    "batch_size": r.batch_size,
                    "success": r.success,
                    "total_seconds": r.total_seconds,
                    "chunks_per_second": r.chunks_per_second,
                    "average_batch_seconds": r.average_batch_seconds,
                    "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
                    "oom_occurred": r.oom_occurred,
                    "oom_retries": r.oom_retries,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def _candidate_batch_sizes(device_info: dict) -> list[int]:
    sizes = list(DEFAULT_BENCHMARK_BATCH_SIZES)
    free_mb = device_info.get("gpu_free_memory_mb")
    total_mb = device_info.get("gpu_total_memory_mb")
    if device_info.get("resolved_device") == "cuda" and free_mb and free_mb > 4500:
        sizes.append(32)
    elif device_info.get("resolved_device") == "cuda" and total_mb and total_mb >= 6000:
        sizes.append(32)
    return sizes


def run_benchmark(
    *,
    config: EmbeddingConfig,
    loader: ChunkLoader,
    chunk_count: int = BENCHMARK_CHUNK_COUNT,
) -> BenchmarkReport:
    device_info = get_device_info(config.device).to_dict()
    chunks = loader.select_deterministic(loader.load_all().chunks, chunk_count)
    texts = [chunk.content for chunk in chunks]

    report = BenchmarkReport(
        device_info=device_info,
        chunk_count=chunk_count,
        model=config.model,
        model_revision=config.model_revision,
        dimension=config.dimension,
    )

    for batch_size in _candidate_batch_sizes(device_info):
        reset_peak_gpu_memory()
        trial_config = replace(config, batch_size=batch_size)
        provider = create_embedding_provider(trial_config)
        runner = BatchEmbeddingRunner(
            provider,
            initial_batch_size=batch_size,
            device_label=device_info.get("resolved_device", config.device),
            gpu_name=device_info.get("gpu_name"),
        )
        started = time.perf_counter()
        try:
            vectors = runner.embed_all(texts)
            elapsed = time.perf_counter() - started
            if len(vectors) != len(texts):
                raise RuntimeError(f"Expected {len(texts)} vectors, got {len(vectors)}")
            batch_count = max(1, (len(texts) + runner.stats.final_batch_size - 1) // runner.stats.final_batch_size)
            report.results.append(
                BenchmarkResult(
                    batch_size=batch_size,
                    success=True,
                    total_seconds=round(elapsed, 3),
                    chunks_per_second=round(len(texts) / max(elapsed, 1e-9), 3),
                    average_batch_seconds=round(elapsed / batch_count, 3),
                    peak_gpu_memory_mb=runner.stats.peak_gpu_memory_mb,
                    oom_occurred=runner.stats.oom_retries > 0,
                    oom_retries=runner.stats.oom_retries,
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            report.results.append(
                BenchmarkResult(
                    batch_size=batch_size,
                    success=False,
                    total_seconds=round(elapsed, 3),
                    chunks_per_second=0.0,
                    average_batch_seconds=0.0,
                    peak_gpu_memory_mb=runner.stats.peak_gpu_memory_mb,
                    oom_occurred=runner.stats.oom_retries > 0 or "oom" in str(exc).lower(),
                    oom_retries=runner.stats.oom_retries,
                    error=str(exc),
                )
            )
    return report
