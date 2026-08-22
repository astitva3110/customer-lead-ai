"""Phase 11 — 100-chunk embedding smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.loader import ChunkLoader
from app.kb.embedding.pipeline import EmbeddingPipeline
from app.kb.embedding.device import get_device_info
from app.kb.vector.search import VectorSearchService
from app.kb.vector.store import VectorStore

SMOKE_CHUNK_COUNT = 100
SMOKE_QUERIES = [
    "Where do I ship returns?",
    "Radius M16 battery life",
    "Return policy refund timeline",
    "FAQ battery life",
    "Prospectus risk factors",
]


def main() -> int:
    config = EmbeddingConfig.from_settings()
    device_info = get_device_info(config.device)
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    store = VectorStore(config.database_url, config.vector_smoke_table, config.dimension)
    store.ensure_schema()

    pipeline = EmbeddingPipeline(config=config, loader=loader, store=store)
    result = pipeline.run(chunk_limit=SMOKE_CHUNK_COUNT, force=True)
    audit = result.audit

    provider = create_embedding_provider(config)
    search = VectorSearchService(config=config, store=store, provider=provider)

    query_checks = []
    for query in SMOKE_QUERIES:
        hits = search.search(query, top_k=5)
        query_checks.append(
            {
                "query": query,
                "hit_count": len(hits),
                "top_chunk_id": hits[0]["chunk_id"] if hits else None,
                "top_document_id": hits[0]["document_id"] if hits else None,
                "top_source_url": hits[0]["source_url"] if hits else None,
                "provenance_ok": bool(hits and hits[0].get("chunk_id") and hits[0].get("source_url")),
            }
        )

    vector_count = store.count(embedding_version=config.embedding_version)
    passed = (
        audit.validated_chunk_count == SMOKE_CHUNK_COUNT
        and audit.embedded_count == SMOKE_CHUNK_COUNT
        and audit.failed_count == 0
        and vector_count == SMOKE_CHUNK_COUNT
        and all(check["provenance_ok"] for check in query_checks if check["hit_count"] > 0)
    )

    report = {
        "smoke_test": "PASS" if passed else "FAIL",
        "device": config.device,
        "device_info": device_info.to_dict(),
        "batch_size": config.batch_size,
        "final_batch_size": audit.final_batch_size,
        "oom_retries": audit.oom_retries,
        "peak_gpu_memory_mb": audit.peak_gpu_memory_mb,
        "smoke_chunk_count": SMOKE_CHUNK_COUNT,
        "validated": audit.validated_chunk_count,
        "embedded": audit.embedded_count,
        "inserted": audit.inserted_count,
        "vector_count": vector_count,
        "vector_dimension": config.dimension,
        "embedding_model": config.model,
        "embedding_model_revision": config.model_revision,
        "embedding_version": config.embedding_version,
        "table": config.vector_smoke_table,
        "query_checks": query_checks,
        "audit": audit.to_dict(),
    }

    json_path = reports_dir / "phase11_smoke_test.json"
    txt_path = reports_dir / "phase11_smoke_test.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(
            [
                "Phase 11 Smoke Test",
                f"SMOKE_TEST: {report['smoke_test']}",
                f"Validated: {audit.validated_chunk_count}/{SMOKE_CHUNK_COUNT}",
                f"Embedded: {audit.embedded_count}/{SMOKE_CHUNK_COUNT}",
                f"Vectors in DB: {vector_count}",
                f"Model: {config.model}",
                f"Revision: {config.model_revision}",
                f"Dimension: {config.dimension}",
                f"Device: {config.device}",
                f"GPU: {device_info.gpu_name or 'n/a'}",
                f"Batch size: {config.batch_size}",
            ]
        ),
        encoding="utf-8",
    )

    print(txt_path.read_text(encoding="utf-8"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
