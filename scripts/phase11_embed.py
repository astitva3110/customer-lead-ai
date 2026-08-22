"""Phase 11 — full production embedding run."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.config import settings
from app.kb.embedding.benchmark import run_benchmark
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.device import get_device_info
from app.kb.embedding.loader import ChunkLoader
from app.kb.embedding.pipeline import EmbeddingPipeline
from app.kb.vector.store import VectorStore


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_production_embed(config: EmbeddingConfig) -> int:
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    manifest_count = len(loader.load_manifest_entries())

    store = VectorStore(config.database_url, config.vector_table, config.dimension)
    store.ensure_schema()

    pipeline = EmbeddingPipeline(config=config, loader=loader, store=store)
    result = pipeline.run(force=False)
    audit = result.audit

    vector_count = store.count(embedding_version=config.embedding_version)
    expected = manifest_count
    passed = (
        audit.failed_count == 0
        and audit.validated_chunk_count == expected
        and vector_count >= expected
        and (audit.embedded_count + audit.skipped_count) == expected
    )

    summary = audit.to_dict()
    summary.update(
        {
            "manifest_chunk_count": manifest_count,
            "vector_count_in_db": vector_count,
            "embedding_run": "PASS" if passed else "FAIL",
        }
    )

    issues = []
    if audit.failed_count:
        issues.append({"type": "embedding_failure", "count": audit.failed_count, "errors": audit.errors})
    if vector_count < expected:
        issues.append(
            {
                "type": "vector_count_mismatch",
                "expected": expected,
                "actual": vector_count,
            }
        )

    (reports_dir / "phase11_embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "phase11_embedding_summary.txt").write_text(
        "\n".join(
            [
                "Phase 11 Embedding Summary",
                f"EMBEDDING_RUN: {summary['embedding_run']}",
                f"Manifest chunks: {manifest_count}",
                f"Validated: {audit.validated_chunk_count}",
                f"Embedded this run: {audit.embedded_count}",
                f"Skipped (already embedded): {audit.skipped_count}",
                f"Failed: {audit.failed_count}",
                f"Vectors in DB: {vector_count}",
                f"Model: {config.model}",
                f"Revision: {config.model_revision}",
                f"Dimension: {config.dimension}",
                f"Device: {audit.device}",
                f"GPU: {audit.gpu_name}",
                f"Batch size: {audit.initial_batch_size} -> {audit.final_batch_size}",
                f"OOM retries: {audit.oom_retries}",
                f"Peak GPU memory MB: {audit.peak_gpu_memory_mb}",
                f"Runtime: {audit.runtime_seconds}s",
                f"Throughput: {audit.throughput_chunks_per_second} chunks/s",
            ]
        ),
        encoding="utf-8",
    )
    (reports_dir / "phase11_embedding_issues.json").write_text(
        json.dumps({"issues": issues, "errors": audit.errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print((reports_dir / "phase11_embedding_summary.txt").read_text(encoding="utf-8"))
    return 0 if passed else 1


def run_benchmark_mode(config: EmbeddingConfig) -> int:
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    report = run_benchmark(config=config, loader=loader)
    report_dict = report.to_dict()
    (reports_dir / "phase11_gpu_benchmark.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "Phase 11 GPU Benchmark",
        f"Device: {report.device_info.get('resolved_device')}",
        f"GPU: {report.device_info.get('gpu_name')}",
        f"Chunks: {report.chunk_count}",
        "",
    ]
    for result in report.results:
        lines.append(
            f"batch_size={result.batch_size} success={result.success} "
            f"rate={result.chunks_per_second} chunks/s peak_mb={result.peak_gpu_memory_mb} "
            f"oom_retries={result.oom_retries}"
        )
    text = "\n".join(lines)
    (reports_dir / "phase11_gpu_benchmark.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if any(r.success for r in report.results) else 1


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Phase 11 embedding pipeline")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 100-chunk GPU benchmark (does not write to production table)",
    )
    args = parser.parse_args()

    config = EmbeddingConfig.from_settings()
    device_info = get_device_info(config.device)
    print(f"Configured device: {config.device}")
    print(f"Detected device info: {json.dumps(device_info.to_dict(), indent=2)}")
    print(f"Batch size: {config.batch_size}")

    if args.benchmark:
        return run_benchmark_mode(config)
    return run_production_embed(config)


if __name__ == "__main__":
    sys.exit(main())
