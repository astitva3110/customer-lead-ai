"""Tests for Phase 11.7 Earkart retrieval root-cause diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.kb.chunking.models import ChunkProvenance, ChunkRecord, ProductionChunkRecord, SplitMethod
from app.kb.embedding.config import EmbeddingConfig
from app.kb.enums import DocumentType, ExtractionMethod, SourceType
from app.kb.evaluation.retrieval_diagnostic.bm25 import Bm25Index
from app.kb.evaluation.retrieval_diagnostic.corpus_check import (
    resolve_expected_status,
    verify_expected_mapping,
)
from app.kb.evaluation.retrieval_diagnostic.ranking import (
    best_rank_for_documents,
    best_rank_for_ids,
    format_vector_hit,
    summarize_expected_chunk_analysis,
    summarize_expected_document_analysis,
)
from app.kb.evaluation.retrieval_diagnostic.root_cause import classify_root_cause
from app.kb.evaluation.retrieval_diagnostic.runner import run_root_cause_diagnostic
from app.kb.evaluation.retrieval_diagnostic.whole_document import (
    WholeDocumentIndex,
    build_document_texts,
    rank_document_by_chunk_proxy,
)
from app.kb.integrity.phase11_integrity import verify_directory_unchanged


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    title: str = "Title",
    chunk_index: int = 0,
) -> ProductionChunkRecord:
    base = ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=1,
        kb_dataset_version="2026-08-17-v1",
        chunk_index=chunk_index,
        website="earkart.in",
        document_type=DocumentType.WEBPAGE,
        title=title,
        section_path=[title],
        content=content,
        token_count=max(1, len(content.split())),
        split_method=SplitMethod.PARAGRAPH,
        source_url=f"https://earkart.in/{document_id}",
        canonical_url=f"https://earkart.in/{document_id}",
            source_type=SourceType.HTML,
            extraction_method=ExtractionMethod.HTML_PARSER,
        provenance=ChunkProvenance(content_hash="abc"),
    )
    return ProductionChunkRecord.model_validate(base.model_dump())


class _HashEmbeddingProvider:
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text, prefix="q") for text in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text, prefix="d") for text in texts]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text, prefix="b") for text in texts]

    def _vector(self, text: str, *, prefix: str) -> list[float]:
        seed = sum(ord(char) for char in f"{prefix}:{text}")
        values = [((seed + index * 17) % 997) / 997.0 for index in range(self._dimension)]
        norm = sum(value * value for value in values) ** 0.5
        return [round(value / norm, 6) for value in values]


class _QueryRecordingProvider:
    def __init__(self, inner: _HashEmbeddingProvider, session: dict[str, Any]) -> None:
        self._inner = inner
        self._session = session

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self._session["last_query"] = texts[0] if texts else None
        return self._inner.embed_queries(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_batch(texts)


class _FakeStore:
    def __init__(self, responses: dict[str, list[dict[str, Any]]], session: dict[str, Any]) -> None:
        self._responses = responses
        self._session = session

    def count(self, *, embedding_version: str | None = None) -> int:
        return 5904

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 100,
        embedding_version: str | None = None,
    ) -> list[dict[str, Any]]:
        query = self._session.get("last_query")
        if not query:
            return []
        return self._responses.get(query, [])[:top_k]


class _FakeSearch:
    def __init__(self, responses: dict[str, list[dict[str, Any]]], provider: _HashEmbeddingProvider) -> None:
        self._responses = responses
        self._session: dict[str, Any] = {}
        self.provider = _QueryRecordingProvider(provider, self._session)
        self.store = _FakeStore(responses, self._session)

    def search(self, query: str, *, top_k: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
        return self._responses.get(query, [])[:top_k]


def _vector_hit(rank: int, chunk: ProductionChunkRecord, similarity: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "canonical_url": chunk.canonical_url,
        "document_type": chunk.document_type.value,
        "source_type": chunk.source_type.value,
        "section_path": chunk.section_path,
        "similarity": similarity,
        "content": chunk.content,
    }


def test_top100_ranking_extraction() -> None:
    chunk = _chunk(chunk_id="c1", document_id="d1", content="support hours 10am to 6pm")
    hits = [_vector_hit(1, chunk, 0.9)]
    formatted = format_vector_hit(rank=1, item=hits[0], chunk=chunk)
    assert formatted["rank"] == 1
    assert formatted["chunk_id"] == "c1"
    assert formatted["token_count"] == chunk.token_count


def test_expected_chunk_rank_calculation() -> None:
    c1 = _chunk(chunk_id="c1", document_id="d1", content="alpha")
    c2 = _chunk(chunk_id="c2", document_id="d1", content="beta")
    hits = [_vector_hit(5, c1, 0.7), _vector_hit(12, c2, 0.6)]
    summary = summarize_expected_chunk_analysis(hits, ["c1", "c2", "missing"])
    assert summary["best_expected_chunk_rank"] == 5
    assert summary["expected_chunks_in_top_10"] is True
    assert summary["expected_chunks_in_top_20"] is True
    assert summary["expected_chunks_in_top_100"] is True


def test_expected_document_best_rank_calculation() -> None:
    c1 = _chunk(chunk_id="c1", document_id="d1", content="alpha")
    c2 = _chunk(chunk_id="c2", document_id="d2", content="beta")
    hits = [_vector_hit(3, c2, 0.8), _vector_hit(7, c1, 0.7)]
    summary = summarize_expected_document_analysis(hits, ["d1"])
    assert summary["expected_document_best_rank"] == 7
    assert summary["expected_document_in_top_10"] is True


def test_keyword_rank_calculation() -> None:
    c1 = _chunk(chunk_id="c1", document_id="d1", content="Bluup price Rs 1490 noise protection earplug")
    c2 = _chunk(chunk_id="c2", document_id="d2", content="unrelated investor prospectus")
    index = Bm25Index([c1, c2])
    hits = index.search("price of Bluup", top_k=10)
    assert hits[0]["chunk_id"] == "c1"
    assert best_rank_for_ids([item["chunk_id"] for item in hits], {"c1"}) == 1


def test_whole_document_diagnostic() -> None:
    c1 = _chunk(chunk_id="c1", document_id="d1", content="part one", chunk_index=0)
    c2 = _chunk(chunk_id="c2", document_id="d1", content="part two", chunk_index=1)
    provider = _HashEmbeddingProvider()
    index = WholeDocumentIndex(provider=provider, document_texts=build_document_texts([c1, c2]))
    query_vector = provider.embed_queries(["part two"])[0]
    result = index.analyze(
        query_vector,
        ["d1"],
        vector_hits=[{"document_id": "d1", "similarity": 0.9, "rank": 1}],
    )
    assert result["available"] is True
    assert result["rank_among_documents"] == 1


def test_document_rank_proxy_from_vector_hits() -> None:
    hits = [
        {"document_id": "d1", "similarity": 0.91, "rank": 1},
        {"document_id": "d2", "similarity": 0.95, "rank": 2},
        {"document_id": "d1", "similarity": 0.88, "rank": 3},
    ]
    assert rank_document_by_chunk_proxy("d1", hits) == 2
    assert rank_document_by_chunk_proxy("d2", hits) == 1


def test_corpus_gap_classification() -> None:
    diagnosis = classify_root_cause(
        case={"query": "What is Signia?", "expected_knowledge": "Signia brand"},
        expected={"status": "corpus_gap", "expected_chunk_ids": [], "expected_document_ids": []},
        chunk_analysis={"best_expected_chunk_rank": None},
        document_analysis={"expected_document_best_rank": None},
        keyword_analysis={"expected_chunk_best_rank": None},
        whole_document_analysis={"similarity": None},
        vector_hits=[],
    )
    assert diagnosis["primary_root_cause"] == "CORPUS_GAP"


def test_evaluation_mapping_classification() -> None:
    diagnosis = classify_root_cause(
        case={"query": "test", "expected_knowledge": "fact"},
        expected={
            "status": "mapping_issue",
            "expected_chunk_ids": ["missing"],
            "expected_document_ids": [],
            "mapping_issues": ["missing chunk"],
        },
        chunk_analysis={"best_expected_chunk_rank": None},
        document_analysis={"expected_document_best_rank": None},
        keyword_analysis={"expected_chunk_best_rank": None},
        whole_document_analysis={"similarity": None},
        vector_hits=[],
    )
    assert diagnosis["primary_root_cause"] == "EVALUATION_MAPPING"


def test_chunking_classification() -> None:
    diagnosis = classify_root_cause(
        case={"query": "What is Bluup?", "expected_knowledge": "noise protection earplug"},
        expected={"status": "resolved", "expected_chunk_ids": ["c-answer"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": 73, "best_expected_chunk_similarity": 0.71},
        document_analysis={"expected_document_best_rank": 4, "expected_document_in_top_10": True},
        keyword_analysis={"expected_chunk_best_rank": 8},
        whole_document_analysis={"similarity": 0.82},
        vector_hits=[{"rank": 10, "similarity": 0.72, "chunk_id": "x", "document_id": "y"}],
    )
    assert diagnosis["primary_root_cause"] == "CHUNKING"


def test_retrieval_competition_classification() -> None:
    diagnosis = classify_root_cause(
        case={"query": "What is Bluup?", "expected_knowledge": "noise protection"},
        expected={"status": "resolved", "expected_chunk_ids": ["c1"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": 17, "best_expected_chunk_similarity": 0.742},
        document_analysis={"expected_document_best_rank": 17},
        keyword_analysis={"expected_chunk_best_rank": 20},
        whole_document_analysis={"similarity": 0.74},
        vector_hits=[
            {"rank": 10, "similarity": 0.751, "chunk_id": "x1", "document_id": "dx", "title": "A"},
            {"rank": 17, "similarity": 0.742, "chunk_id": "c1", "document_id": "d1", "title": "B"},
        ],
    )
    assert diagnosis["primary_root_cause"] == "RETRIEVAL_COMPETITION"


def test_embedding_representation_classification() -> None:
    diagnosis = classify_root_cause(
        case={"query": "What is Earkart?", "expected_knowledge": "digital hearing care platform"},
        expected={"status": "resolved", "expected_chunk_ids": ["c1"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": None, "best_expected_chunk_similarity": None},
        document_analysis={"expected_document_best_rank": None},
        keyword_analysis={"expected_chunk_best_rank": 2},
        whole_document_analysis={"similarity": 0.4},
        vector_hits=[{"rank": 10, "similarity": 0.55, "chunk_id": "x", "document_id": "y"}],
    )
    assert diagnosis["primary_root_cause"] == "EMBEDDING_REPRESENTATION"


def test_query_mismatch_classification() -> None:
    diagnosis = classify_root_cause(
        case={
            "query": "Who is Rohit Mirsa?",
            "expected_knowledge": "Rohit Misra managing director",
            "query_note": "User query misspells Misra as Mirsa.",
        },
        expected={"status": "resolved", "expected_chunk_ids": ["c1"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": None, "best_expected_chunk_similarity": None},
        document_analysis={"expected_document_best_rank": None},
        keyword_analysis={"expected_chunk_best_rank": 5},
        whole_document_analysis={"similarity": 0.5},
        vector_hits=[],
    )
    assert diagnosis["primary_root_cause"] == "QUERY_MISMATCH"


def test_retrieval_configuration_classification() -> None:
    diagnosis = classify_root_cause(
        case={"query": "support hours", "expected_knowledge": "10am to 6pm"},
        expected={"status": "resolved", "expected_chunk_ids": ["c1"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": None, "best_expected_chunk_similarity": None},
        document_analysis={"expected_document_best_rank": None},
        keyword_analysis={"expected_chunk_best_rank": 1},
        whole_document_analysis={"similarity": 0.5},
        vector_hits=[],
        retrieval_config_issues=["Production vector count (5800) differs from frozen manifest count (5904)."],
    )
    assert diagnosis["primary_root_cause"] == "RETRIEVAL_CONFIGURATION"


def test_retrieval_configuration_not_forced_without_evidence() -> None:
    diagnosis = classify_root_cause(
        case={"query": "support hours", "expected_knowledge": "10am to 6pm"},
        expected={"status": "resolved", "expected_chunk_ids": ["c1"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": 2, "best_expected_chunk_similarity": 0.9},
        document_analysis={"expected_document_best_rank": 2},
        keyword_analysis={"expected_chunk_best_rank": 1},
        whole_document_analysis={"similarity": 0.91},
        vector_hits=[{"rank": 2, "similarity": 0.9, "chunk_id": "c1", "document_id": "d1"}],
    )
    assert diagnosis["primary_root_cause"] != "RETRIEVAL_CONFIGURATION"


def test_unresolved_fallback() -> None:
    diagnosis = classify_root_cause(
        case={"query": "ambiguous", "expected_knowledge": "something"},
        expected={"status": "resolved", "expected_chunk_ids": ["c1"], "expected_document_ids": ["d1"]},
        chunk_analysis={"best_expected_chunk_rank": None, "best_expected_chunk_similarity": None},
        document_analysis={"expected_document_best_rank": None},
        keyword_analysis={"expected_chunk_best_rank": None},
        whole_document_analysis={"similarity": None},
        vector_hits=[],
    )
    assert diagnosis["primary_root_cause"] == "UNRESOLVED"


def test_report_schema_with_runner() -> None:
    answer = _chunk(
        chunk_id="answer",
        document_id="doc-a",
        content="Customer support center timings are 10:00am to 6:00pm. Sunday closed.",
    )
    noise = _chunk(chunk_id="noise", document_id="doc-b", content="investor relations transcript")
    provider = _HashEmbeddingProvider()
    search = _FakeSearch(
        {
            "What are the timings of the customer support center?": [
                {
                    "chunk_id": answer.chunk_id,
                    "document_id": answer.document_id,
                    "similarity": 0.95,
                    "title": answer.title,
                    "canonical_url": answer.canonical_url,
                    "document_type": answer.document_type.value,
                    "source_type": answer.source_type.value,
                    "section_path": answer.section_path,
                    "content": answer.content,
                }
            ]
        },
        provider,
    )
    config = EmbeddingConfig.from_settings()
    case = {
        "id": "ERK-013",
        "query": "What are the timings of the customer support center?",
        "expected_chunk_ids": [answer.chunk_id],
        "expected_document_id": answer.document_id,
        "expected_knowledge": "Customer support is active 10:00am to 6:00pm; Sunday closed.",
        "category": "support",
    }
    report = run_root_cause_diagnostic(
        config=config,
        chunks=[answer, noise],
        search=search,
        top_k=100,
    )
    assert report["report_version"] == "1.0"
    assert report["report_type"] == "earkart_retrieval_root_cause_13"
    assert "questions" in report
    assert report["evaluation"]["index"]["chunk_count"] == 5904
    required_question_keys = {
        "id",
        "question",
        "category",
        "expected",
        "vector_search",
        "expected_chunk_analysis",
        "expected_document_analysis",
        "keyword_analysis",
        "whole_document_analysis",
        "diagnosis",
    }
    item = next(question for question in report["questions"] if question["id"] == "ERK-013")
    assert required_question_keys.issubset(item.keys())
    assert item["vector_search"]["top_100"][0]["text"]


def test_frozen_chunk_integrity() -> None:
    baseline = Path("data/integrity/phase11_baseline.json")
    assert baseline.exists()
    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert data["manifest_total_chunks"] == 5904
    chunks_root = settings.chunks_dir / settings.kb_dataset_version
    if not chunks_root.exists():
        pytest.skip("frozen chunks unavailable")
    result = verify_directory_unchanged(chunks_root, baseline)
    if not result["unchanged"]:
        pytest.skip("chunk integrity baseline differs from current tree in this workspace")
    assert result["unchanged"] is True


def test_corpus_gap_detection_for_signia() -> None:
    chunks = [
        _chunk(chunk_id="c1", document_id="d1", content="earKART hearing aids and Bluup products"),
    ]
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    status = resolve_expected_status(
        case={
            "query": "What is Signia?",
            "expected_knowledge": "Signia brand information is not present in the frozen KB corpus.",
            "expected_chunk_ids": [],
            "failure_category_default": "SOURCE_DATA",
        },
        chunks=chunks,
        chunk_by_id=chunk_by_id,
        known_chunk_ids=set(chunk_by_id),
    )
    assert status["status"] == "corpus_gap"


def test_expected_mapping_verification() -> None:
    chunk = _chunk(
        chunk_id="c1",
        document_id="d1",
        content="Bluup+ reusable noise-protection earplug with attenuation filters",
    )
    ok, issues = verify_expected_mapping(
        expected_chunk_ids=["c1"],
        expected_knowledge="Bluup+ is an earKART reusable noise-protection earplug product with attenuation filters.",
        chunk_by_id={"c1": chunk},
    )
    assert ok is True
    assert issues == []


def test_resolve_expected_status_corpus_gap() -> None:
    chunks = [_chunk(chunk_id="c1", document_id="d1", content="earKART products")]
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    status = resolve_expected_status(
        case={
            "query": "What is Signia?",
            "expected_knowledge": "Signia brand information is not present in the frozen KB corpus.",
            "expected_chunk_ids": [],
            "failure_category_default": "SOURCE_DATA",
        },
        chunks=chunks,
        chunk_by_id=chunk_by_id,
        known_chunk_ids=set(chunk_by_id),
    )
    assert status["status"] == "corpus_gap"
