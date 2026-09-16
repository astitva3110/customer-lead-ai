"""Production retrieval configuration — single source of truth for chat + ingestion."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.ingestion.indexing import Phase12IndexIdentity, Phase12VectorStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RETRIEVAL_CONFIG = _REPO_ROOT / "configs" / "retrieval" / "v3_2_pgvector.yaml"


def _resolve_config_path() -> Path:
    configured = settings.retrieval_config_path
    path = configured if configured.is_absolute() else _REPO_ROOT / configured
    if path.exists():
        return path
    if _DEFAULT_RETRIEVAL_CONFIG.exists():
        return _DEFAULT_RETRIEVAL_CONFIG
    return path


def load_runtime_retrieval_config() -> RetrievalConfig:
    """Load retrieval settings for runtime chat and admin ingestion.

    YAML supplies corpus metadata; live identity fields come from ``settings`` so
    ``PHASE12_VECTOR_TABLE``, ``EMBEDDING_VERSION``, etc. can be changed via ``.env``
    without editing Python or re-copying evaluation corpora.
    """
    base = RetrievalConfig.from_yaml(_resolve_config_path())
    table = (settings.phase12_vector_table or base.vector_table or "").strip()
    return RetrievalConfig(
        corpus_version=base.corpus_version,
        vector_table=table or None,
        embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_model_revision,
        embedding_version=settings.embedding_version,
        kb_dataset_version=settings.kb_dataset_version,
        chunking_algorithm_version=settings.chunking_algorithm_version,
        embedding_input_manifest=settings.embedding_input_manifest,
        top_k=base.top_k,
        mode=base.mode,
        chunks_path=base.chunks_path,
    )


def build_production_vector_store(config: RetrievalConfig | None = None) -> Phase12VectorStore:
    resolved = config or load_runtime_retrieval_config()
    if not resolved.vector_table:
        raise ValueError("vector_table is required for production PGVector retrieval")
    return Phase12VectorStore(
        table_name=resolved.vector_table,
        identity=Phase12IndexIdentity.from_retrieval_config(resolved),
    )


def ensure_production_vector_store_schema() -> RetrievalConfig:
    """Create the configured PGVector table/indexes without loading evaluation data."""
    config = load_runtime_retrieval_config()
    if config.mode != "pgvector" or not config.vector_table:
        return config
    store = build_production_vector_store(config)
    store.ensure_schema()
    store.ensure_content_fts_index()
    return config
