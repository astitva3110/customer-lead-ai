from __future__ import annotations

import uuid

import pytest

from app.db.engine import ensure_schema
from app.kb.retrieval.runtime_config import (
    build_production_vector_store,
    load_runtime_retrieval_config,
)


def _require_postgres() -> None:
    try:
        ensure_schema()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")


@pytest.mark.integration
def test_fresh_vector_table_exists_with_zero_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_postgres()
    table_name = f"chunk_embeddings_test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr("app.config.settings.phase12_vector_table", table_name)

    config = load_runtime_retrieval_config()
    assert config.vector_table == table_name

    store = build_production_vector_store(config)
    store.ensure_schema()
    store.ensure_content_fts_index()

    assert store.count() == 0
    assert store.search([0.0] * store.dimension, top_k=5) == []
    assert store.search_keyword("hearing aid", top_k=5) == []


@pytest.mark.integration
def test_empty_vector_table_keyword_search_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_postgres()
    table_name = f"chunk_embeddings_test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr("app.config.settings.phase12_vector_table", table_name)

    store = build_production_vector_store()
    store.ensure_schema()
    store.ensure_content_fts_index()

    hits = store.search_keyword("What is TINY?", top_k=10)
    assert hits == []
