#!/usr/bin/env python3
"""Phase 14 — KB V2 smoke embedding against Phase 13 validated corpus."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 14 KB V2 smoke embedding")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.environ["EMBEDDING_DEVICE"] = args.device
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from app.config import settings
    from app.kb.embedding.device import get_device_info
    from app.kb.embedding.phase13_corpus import load_phase13_validated_corpus, select_deterministic_chunks
    from app.kb.embedding.phase14_pipeline import Phase14EmbeddingPipeline
    from app.kb.embedding.phase14_verify import format_final_summary, verify_smoke_checks
    from app.kb.ingestion.indexing import Phase12VectorStore

    device_info = get_device_info(args.device)
    corpus = load_phase13_validated_corpus()
    smoke_chunks = select_deterministic_chunks(corpus.all_chunks, settings.phase14_smoke_chunk_count)

    pipeline = Phase14EmbeddingPipeline(
        device=args.device,
        batch_size=args.batch_size,
        table_name=settings.phase14_vector_smoke_table,
    )
    result = pipeline.run(corpus, smoke_chunks, force=True)
    store = Phase12VectorStore(table_name=settings.phase14_vector_smoke_table)
    vector_count = store.count(embedding_version=settings.phase12_embedding_version)

    verification = verify_smoke_checks(
        corpus=corpus,
        chunks=smoke_chunks,
        store=store,
        v1_before=result.v1_vector_count_before,
        v1_after=result.v1_vector_count_after,
        embedded_count=result.audit.embedded_count,
        failed_count=result.audit.failed_count,
    )

    passed = (
        verification["pass"]
        and result.audit.failed_count == 0
        and result.audit.embedded_count == len(smoke_chunks)
        and vector_count == len(smoke_chunks)
    )

    report = {
        "smoke_test": "PASS" if passed else "FAIL",
        "smoke_chunk_count": len(smoke_chunks),
        "corpus_chunk_count": corpus.chunk_count,
        "source_pdfs": corpus.source_pdfs,
        "validated": result.audit.validated_chunk_count,
        "embedded": result.audit.embedded_count,
        "inserted": result.audit.inserted_count,
        "skipped": result.audit.skipped_count,
        "failed": result.audit.failed_count,
        "vector_count": vector_count,
        "vector_dimension": settings.embedding_dimension,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_version": settings.phase12_embedding_version,
        "embedding_input_manifest": settings.phase12_embedding_input_manifest,
        "chunking_algorithm_version": settings.phase12_chunking_algorithm_version,
        "kb_dataset_version": settings.phase12_kb_dataset_version,
        "table": settings.phase14_vector_smoke_table,
        "device": device_info.resolved_device,
        "gpu": device_info.gpu_name,
        "device_info": device_info.to_dict(),
        "verification": verification,
        "audit": result.audit.to_dict(),
        "v1_vector_count_before": result.v1_vector_count_before,
        "v1_vector_count_after": result.v1_vector_count_after,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase14_embedding_smoke.json"
    txt_path = reports_dir / "phase14_embedding_smoke.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = format_final_summary(
        passed=passed,
        source_chunks=corpus.chunk_count,
        validated=result.audit.validated_chunk_count,
        embedded=result.audit.embedded_count,
        failed=result.audit.failed_count,
        skipped=result.audit.skipped_count,
        device=device_info.resolved_device,
        gpu=device_info.gpu_name,
        v1_modified=result.v1_vector_count_before != result.v1_vector_count_after,
    )
    txt_path.write_text(
        "\n".join(
            [
                "Phase 14 KB V2 Smoke Embedding",
                f"SMOKE_TEST: {'PASS' if passed else 'FAIL'}",
                f"Smoke chunks: {len(smoke_chunks)}/{corpus.chunk_count}",
                f"Table: {settings.phase14_vector_smoke_table}",
                "",
                summary,
            ]
        ),
        encoding="utf-8",
    )

    print(txt_path.read_text(encoding="utf-8"))
    print(f"\nWrote {json_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
