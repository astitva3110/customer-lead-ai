"""Phase 14 embedding verification helpers."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.kb.embedding.phase13_corpus import Phase13Corpus
from app.kb.ingestion.indexing import Phase12VectorStore
from app.kb.ingestion.models import Phase12ChunkRecord


def verify_smoke_checks(
    *,
    corpus: Phase13Corpus,
    chunks: list[Phase12ChunkRecord],
    store: Phase12VectorStore,
    v1_before: int,
    v1_after: int,
    embedded_count: int,
    failed_count: int,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["phase13_source"] = {
        "pass": len(corpus.source_pdfs) == 2 and corpus.chunk_count > 0,
        "source_pdfs": corpus.source_pdfs,
        "corpus_chunk_count": corpus.chunk_count,
        "kb_dataset_version": settings.phase12_kb_dataset_version,
    }

    sample = chunks[0] if chunks else None
    row = store.get_by_chunk_id(sample.chunk_id) if sample else None

    checks["embedding_succeeded"] = {
        "pass": embedded_count == len(chunks) and failed_count == 0,
        "embedded_count": embedded_count,
        "expected": len(chunks),
        "failed_count": failed_count,
    }

    checks["vector_dimension"] = {
        "pass": row is not None and row.get("embedding_dimension") == 1024,
        "expected": 1024,
        "actual": row.get("embedding_dimension") if row else None,
    }

    checks["embedding_model"] = {
        "pass": row is not None and row.get("embedding_model") == settings.embedding_model,
        "expected": settings.embedding_model,
        "actual": row.get("embedding_model") if row else None,
    }

    checks["embedding_revision"] = {
        "pass": row is not None and row.get("embedding_model_revision") == settings.embedding_model_revision,
        "expected": settings.embedding_model_revision,
        "actual": row.get("embedding_model_revision") if row else None,
    }

    checks["chunk_id_preserved"] = {
        "pass": row is not None and sample is not None and row.get("chunk_id") == sample.chunk_id,
        "chunk_id": sample.chunk_id if sample else None,
    }

    checks["document_id_preserved"] = {
        "pass": row is not None and sample is not None and row.get("document_id") == sample.document_id,
        "document_id": sample.document_id if sample else None,
    }

    checks["chunking_algorithm_version"] = {
        "pass": row is not None
        and row.get("chunking_algorithm_version") == settings.phase12_chunking_algorithm_version,
        "expected": settings.phase12_chunking_algorithm_version,
        "actual": row.get("chunking_algorithm_version") if row else None,
    }

    checks["embedding_input_manifest"] = {
        "pass": row is not None
        and row.get("embedding_input_manifest") == settings.phase12_embedding_input_manifest,
        "expected": settings.phase12_embedding_input_manifest,
        "actual": row.get("embedding_input_manifest") if row else None,
    }

    checks["old_kb_unmodified"] = {
        "pass": v1_before == v1_after,
        "v1_before": v1_before,
        "v1_after": v1_after,
    }

    checks["kb_v2_table_only"] = {
        "pass": store.table_name in {settings.phase12_vector_table, settings.phase14_vector_smoke_table},
        "table": store.table_name,
    }

    all_pass = all(item.get("pass") for item in checks.values())
    return {"pass": all_pass, "checks": checks}


def format_final_summary(
    *,
    passed: bool,
    source_chunks: int,
    validated: int,
    embedded: int,
    failed: int,
    skipped: int,
    device: str,
    gpu: str | None,
    v1_modified: bool,
) -> str:
    return "\n".join(
        [
            "PHASE14_EMBEDDING: PASS" if passed else "PHASE14_EMBEDDING: FAIL",
            f"SOURCE_CHUNKS: {source_chunks}",
            f"VALIDATED: {validated}",
            f"EMBEDDED: {embedded}",
            f"FAILED: {failed}",
            f"SKIPPED: {skipped}",
            "VECTOR_DIMENSION: 1024",
            f"MODEL: {settings.embedding_model}",
            f"REVISION: {settings.embedding_model_revision}",
            f"DEVICE: {device}",
            f"GPU: {gpu or 'n/a'}",
            f"OLD_5904_VECTORS_MODIFIED: {'YES' if v1_modified else 'NO'}",
        ]
    )
