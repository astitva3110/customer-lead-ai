#!/usr/bin/env python3
"""Phase 14 — full KB V2 production embedding of Phase 13 validated corpus."""

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
    parser = argparse.ArgumentParser(description="Phase 14 KB V2 production embedding")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-smoke-check", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-embed even if vectors exist")
    args = parser.parse_args()

    os.environ["EMBEDDING_DEVICE"] = args.device
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from app.config import settings
    from app.kb.embedding.device import get_device_info
    from app.kb.embedding.phase13_corpus import load_phase13_validated_corpus
    from app.kb.embedding.phase14_pipeline import Phase14EmbeddingPipeline
    from app.kb.embedding.phase14_verify import format_final_summary
    from app.kb.ingestion.indexing import Phase12VectorStore

    smoke_path = settings.reports_dir / "phase14_embedding_smoke.json"
    if not args.skip_smoke_check:
        if not smoke_path.exists():
            print("Smoke report missing. Run scripts/phase14_embed_smoke.py first.", file=sys.stderr)
            return 1
        smoke_report = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_report.get("smoke_test") != "PASS":
            print("Smoke test did not pass. Aborting full embedding.", file=sys.stderr)
            return 1

    device_info = get_device_info(args.device)
    corpus = load_phase13_validated_corpus()
    all_chunks = corpus.all_chunks

    pipeline = Phase14EmbeddingPipeline(
        device=args.device,
        batch_size=args.batch_size,
        table_name=settings.phase12_vector_table,
    )
    result = pipeline.run(corpus, all_chunks, force=args.force)
    store = Phase12VectorStore(table_name=settings.phase12_vector_table)
    vector_count = store.count(embedding_version=settings.phase12_embedding_version)

    passed = (
        result.audit.failed_count == 0
        and result.audit.validated_chunk_count == corpus.chunk_count
        and (result.audit.embedded_count + result.audit.skipped_count) == corpus.chunk_count
        and vector_count >= corpus.chunk_count
        and result.v1_vector_count_before == result.v1_vector_count_after
    )

    report = {
        "embedding_run": "PASS" if passed else "FAIL",
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
        "table": settings.phase12_vector_table,
        "device": device_info.resolved_device,
        "gpu": device_info.gpu_name,
        "device_info": device_info.to_dict(),
        "audit": result.audit.to_dict(),
        "v1_vector_count_before": result.v1_vector_count_before,
        "v1_vector_count_after": result.v1_vector_count_after,
        "old_5904_vectors_modified": result.v1_vector_count_before != result.v1_vector_count_after,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase14_embedding_summary.json"
    txt_path = reports_dir / "phase14_embedding_summary.txt"
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
                "Phase 14 KB V2 Production Embedding",
                f"Table: {settings.phase12_vector_table}",
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
