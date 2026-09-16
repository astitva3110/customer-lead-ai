from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.retrieval.runtime_config import (
    build_production_vector_store,
    load_runtime_retrieval_config,
)


def test_load_runtime_retrieval_config_uses_settings_over_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "phase12_vector_table", "chunk_embeddings_custom")
    monkeypatch.setattr(settings, "embedding_version", "embed_custom_v1")
    monkeypatch.setattr(settings, "kb_dataset_version", "dataset-custom")
    monkeypatch.setattr(settings, "chunking_algorithm_version", "chunk-custom")
    monkeypatch.setattr(settings, "embedding_input_manifest", "manifest-custom")

    config = load_runtime_retrieval_config()

    assert config.vector_table == "chunk_embeddings_custom"
    assert config.embedding_version == "embed_custom_v1"
    assert config.kb_dataset_version == "dataset-custom"
    assert config.chunking_algorithm_version == "chunk-custom"
    assert config.embedding_input_manifest == "manifest-custom"
    assert config.mode == "pgvector"


def test_build_production_vector_store_uses_configured_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "phase12_vector_table", "chunk_embeddings_test_cfg")

    config = RetrievalConfig(
        corpus_version="v-test",
        vector_table="chunk_embeddings_test_cfg",
        embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_model_revision,
        embedding_version=settings.embedding_version,
        kb_dataset_version=settings.kb_dataset_version,
        chunking_algorithm_version=settings.chunking_algorithm_version,
        embedding_input_manifest=settings.embedding_input_manifest,
    )
    store = build_production_vector_store(config)

    assert store.table_name == "chunk_embeddings_test_cfg"
    assert store.identity.embedding_version == settings.embedding_version


def test_retrieval_config_path_is_relative_to_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "custom_retrieval.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "corpus_version: v-custom",
                "vector_table: chunk_embeddings_yaml",
                "mode: pgvector",
                "embedding_version: yaml_embed_v1",
                "kb_dataset_version: yaml_kb",
                "chunking_algorithm_version: yaml_chunk",
                "embedding_input_manifest: yaml_manifest",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "retrieval_config_path", yaml_path)
    monkeypatch.setattr(settings, "phase12_vector_table", "")

    config = load_runtime_retrieval_config()

    assert config.corpus_version == "v-custom"
    assert config.vector_table == "chunk_embeddings_yaml"
