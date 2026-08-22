"""Phase 11 embedding, vector, and evaluation tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.kb.chunking.models import ChunkProvenance, ChunkRecord, ProductionChunkRecord, SplitMethod
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.embedding.pipeline import EmbeddingPipeline
from app.kb.embedding.validation import (
    ChunkValidationError,
    reject_non_embed_ready_status,
    validate_production_chunk,
)
from app.kb.embedding.versioning import VectorIndexIdentity
from app.kb.enums import DocumentType, ExtractionMethod, SourceType
from app.kb.chunking.question_coverage import QUESTION_COVERAGE
from app.kb.evaluation.dataset import build_evaluation_dataset, resolve_expected_chunk_ids
from app.kb.evaluation.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from app.kb.evaluation.regression import REGRESSION_QUERIES
from app.kb.vector.store import VectorRecordInput, VectorStore

KB_VERSION = "2026-08-17-v1"


class MockEmbeddingProvider:
    provider_name = "mock"
    model_name = "mock-model"
    model_revision = "mock-rev"
    dimension = 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(f"query:{text}") for text in texts]

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) % 997
        rng = np.random.default_rng(seed)
        vec = rng.random(self.dimension)
        vec = vec / np.linalg.norm(vec)
        return vec.astype(float).tolist()


def _sample_chunk(**overrides) -> ProductionChunkRecord:
    base = ProductionChunkRecord(
        chunk_id="abc123",
        document_id="doc-1",
        document_version=1,
        kb_dataset_version=KB_VERSION,
        chunk_index=0,
        website="earkart.com",
        document_type=DocumentType.POLICY,
        title="Test",
        section_path=["Test"],
        content="Sample policy content",
        token_count=10,
        split_method=SplitMethod.PARAGRAPH,
        source_url="https://earkart.com/pages/returns",
        canonical_url="https://earkart.com/pages/returns",
        page_number=None,
        source_type=SourceType.HTML,
        extraction_method=ExtractionMethod.HTML_PARSER,
        provenance=ChunkProvenance(content_hash="sha256:" + "a" * 64),
        eligibility_status="EMBED_READY",
    )
    data = base.model_dump()
    data.update(overrides)
    return ProductionChunkRecord(**data)


def test_embedding_config_default_batch_size_eight() -> None:
    config = EmbeddingConfig.from_settings()
    assert config.batch_size == 8


def test_embedding_config_from_settings() -> None:
    config = EmbeddingConfig.from_settings()
    assert config.model == "Qwen/Qwen3-Embedding-0.6B"
    assert config.dimension == 1024
    assert config.chunking_algorithm_version == "phase10.6"
    assert config.embedding_input_manifest == KB_VERSION


def test_validate_production_chunk_rejects_empty_content() -> None:
    with pytest.raises(ChunkValidationError):
        validate_production_chunk(_sample_chunk(content="   "))


def test_validate_production_chunk_rejects_non_embed_ready() -> None:
    with pytest.raises(ChunkValidationError):
        validate_production_chunk(_sample_chunk(eligibility_status="REVIEW"))


def test_reject_review_and_do_not_embed() -> None:
    with pytest.raises(ChunkValidationError):
        reject_non_embed_ready_status("REVIEW", chunk_id="x")
    with pytest.raises(ChunkValidationError):
        reject_non_embed_ready_status("DO_NOT_EMBED", chunk_id="x")


def test_chunk_loader_reads_frozen_manifest() -> None:
    loader = ChunkLoader(settings.chunks_dir, KB_VERSION)
    result = loader.load_all()
    assert result.manifest_count == 5904
    assert len(result.chunks) == 5904
    assert all(c.eligibility_status == "EMBED_READY" for c in result.chunks)


def test_deterministic_smoke_selection() -> None:
    loader = ChunkLoader(settings.chunks_dir, KB_VERSION)
    chunks = loader.load_all().chunks
    first = loader.select_deterministic(chunks, 100)
    second = loader.select_deterministic(chunks, 100)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len(first) == 100


def test_recall_metrics() -> None:
    assert recall_at_k(["a"], ["a", "b"], 1) == 1.0
    assert recall_at_k(["a"], ["b", "a"], 1) == 0.0
    assert recall_at_k(["a"], ["b", "a"], 2) == 1.0
    assert reciprocal_rank(["a"], ["b", "a"]) == 0.5
    metrics = aggregate_metrics(
        [
            {"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "passed": True},
            {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 0.25, "passed": True},
        ]
    )
    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr"] == 0.625


def test_evaluation_dataset_resolves_qc08_chunk() -> None:
    loader = ChunkLoader(settings.chunks_dir, KB_VERSION)
    chunks = loader.load_all().chunks
    dataset = build_evaluation_dataset(chunks)
    qc08 = next(item for item in dataset if item["id"] == "QC08")
    assert qc08["expected_chunk_ids"]
    chunk = next(c for c in chunks if c.chunk_id == qc08["expected_chunk_ids"][0])
    assert "3.3" in chunk.content
    assert "Sector 62" in chunk.content


def test_evaluation_dataset_resolves_qc25_multi_chunk() -> None:
    loader = ChunkLoader(settings.chunks_dir, KB_VERSION)
    chunks = loader.load_all().chunks
    qc25 = next(q for q in QUESTION_COVERAGE if q["id"] == "QC25")
    expected_ids, require_all = resolve_expected_chunk_ids(chunks, qc25)
    assert len(expected_ids) >= 2
    assert require_all is True


def test_vector_record_input_preserves_metadata() -> None:
    chunk = _sample_chunk()
    config = EmbeddingConfig.from_settings()
    identity = VectorIndexIdentity.from_config(config)
    record = VectorRecordInput.from_chunk(
        chunk=chunk,
        embedding=[0.1] * 1024,
        identity=identity,
        provider_name="mock",
    )
    assert record.chunk_id == chunk.chunk_id
    assert record.source_url == chunk.source_url
    assert record.content == chunk.content


def test_version_identity_unique_key() -> None:
    config = EmbeddingConfig.from_settings()
    identity = VectorIndexIdentity.from_config(config)
    assert "phase10.6" in identity.unique_key
    assert "Qwen/Qwen3-Embedding-0.6B" in identity.unique_key


@pytest.mark.integration
def test_pgvector_insert_and_search() -> None:
    config = EmbeddingConfig.from_settings()
    store = VectorStore(config.database_url, "chunk_embeddings_test", config.dimension)
    try:
        store.ensure_schema()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")

    chunk = _sample_chunk(chunk_id="test-chunk-integration")
    identity = VectorIndexIdentity.from_config(config)
    provider = MockEmbeddingProvider()
    vector = provider.embed_documents([chunk.content])[0]
    record = VectorRecordInput.from_chunk(
        chunk=chunk,
        embedding=vector,
        identity=identity,
        provider_name="mock",
    )
    inserted = store.upsert_many([record], force=True)
    assert inserted == 1
    results = store.search(vector, top_k=1, embedding_version=config.embedding_version)
    assert results
    assert results[0]["chunk_id"] == chunk.chunk_id


def test_mock_pipeline_smoke_batch() -> None:
    config = EmbeddingConfig.from_settings()
    loader = ChunkLoader(settings.chunks_dir, KB_VERSION)
    try:
        store = VectorStore(config.database_url, "chunk_embeddings_pipeline_test", config.dimension)
        store.ensure_schema()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")

    pipeline = EmbeddingPipeline(
        config=config,
        loader=loader,
        store=store,
        provider=MockEmbeddingProvider(),
    )
    result = pipeline.run(chunk_limit=5, force=True)
    assert result.audit.validated_chunk_count == 5
    assert result.audit.embedded_count == 5
    assert result.audit.failed_count == 0


def test_regression_query_definitions_exist() -> None:
    assert any(q["id"] == "REG_PRODUCT_BATTERY_LIFE" for q in REGRESSION_QUERIES)
    assert any("270 Hrs" in needle for q in REGRESSION_QUERIES for needle in q["needles"])


def test_integrity_baseline_exists() -> None:
    baseline = Path("data/integrity/phase11_baseline.json")
    assert baseline.exists()
    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert data["manifest_total_chunks"] == 5904
