from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.kb.evaluation.corpus_registry import load_corpus
from app.kb.ingestion.phase16_corpus_v3_2 import build_v3_2_integrity_report
from app.kb.vector.store import VectorStore

V1_MANIFEST = Path("data/chunks/2026-08-17-v1/manifest.json")


def test_v3_2_chunk_count_unchanged() -> None:
    integrity = build_v3_2_integrity_report()
    corpus = load_corpus("v3.2")
    assert integrity.passed
    assert integrity.v3_2_total_chunks == 127
    assert integrity.changed_existing_chunks == 0
    assert corpus.metadata["chunk_count"] == 127


def test_v3_2_covers_listed_knowledge_topics() -> None:
    corpus = load_corpus("v3.2")
    blob = "\n".join(chunk.content for chunk in corpus.chunks).lower()
    for needle in ("tiny", "bte", "bluup", "warranty", "delivery", "earkart"):
        assert needle in blob, f"V3.2 corpus missing {needle}"


def test_v1_manifest_is_frozen() -> None:
    payload = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    assert payload["kb_dataset_version"] == "2026-08-17-v1"
    assert payload["total_chunks"] == 5904


def test_v1_pgvector_count_if_available() -> None:
    store = VectorStore(settings.database_url, settings.vector_table, settings.embedding_dimension)
    try:
        count = store.count(embedding_version=settings.embedding_version)
    except Exception as exc:
        import pytest

        pytest.skip(f"V1 PGVector unavailable: {exc}")
    assert count == 5904
    assert count > 0
