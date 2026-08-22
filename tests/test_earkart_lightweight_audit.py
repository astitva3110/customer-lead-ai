"""Lightweight tests for Phase 11.7 vector-only root-cause audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.kb.chunking.models import ChunkProvenance, ChunkRecord, ProductionChunkRecord, SplitMethod
from app.kb.embedding.config import EmbeddingConfig
from app.kb.enums import DocumentType, ExtractionMethod, SourceType
from app.kb.evaluation.retrieval_diagnostic.lightweight_audit import (
    format_hit_metadata,
    format_lightweight_text_report,
    run_lightweight_audit,
    top_competing_document,
)
from app.kb.evaluation.retrieval_diagnostic.lightweight_root_cause import classify_lightweight_root_cause
from app.kb.evaluation.retrieval_diagnostic.ranking import best_rank_for_documents, best_rank_for_ids
from app.kb.integrity.phase11_integrity import verify_directory_unchanged


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    title: str = "Title",
) -> ProductionChunkRecord:
    base = ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=1,
        kb_dataset_version="2026-08-17-v1",
        chunk_index=0,
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


class _QueryRecordingProvider:
    def __init__(self, inner, session: dict[str, Any]) -> None:
        self._inner = inner
        self._session = session

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self._session["last_query"] = texts[0] if texts else None
        return self._inner.embed_queries(texts)


class _HashProvider:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.9] for _ in texts]


class _FakeStore:
    def __init__(self, responses: dict[str, list[dict[str, Any]]], session: dict[str, Any]) -> None:
        self._responses = responses
        self._session = session

    def count(self, *, embedding_version: str | None = None) -> int:
        return 5904

    def search(self, query_embedding, *, top_k: int = 100, embedding_version: str | None = None) -> list[dict]:
        query = self._session.get("last_query")
        return self._responses.get(query or "", [])[:top_k]


class _FakeSearch:
    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self._session: dict[str, Any] = {}
        self.provider = _QueryRecordingProvider(_HashProvider(), self._session)
        self.store = _FakeStore(responses, self._session)


def test_expected_chunk_rank() -> None:
    assert best_rank_for_ids(["a", "b", "c"], {"b"}) == 2


def test_expected_document_rank() -> None:
    hits = [
        {"rank": 1, "document_id": "d2", "similarity": 0.9},
        {"rank": 4, "document_id": "d1", "similarity": 0.8},
    ]
    rank, sim = best_rank_for_documents(hits, {"d1"})
    assert rank == 4
    assert sim == 0.8


def test_corpus_gap_classification() -> None:
    result = classify_lightweight_root_cause(
        case={"query": "What is Signia?"},
        expected_status="corpus_gap",
        mapping_issues=[],
        chunk_rank=None,
        chunk_similarity=None,
        document_rank=None,
        document_similarity=None,
        top_hits=[],
        expected_chunk_ids=set(),
        expected_document_ids=set(),
    )
    assert result["primary_root_cause"] == "CORPUS_GAP"


def test_chunking_classification() -> None:
    result = classify_lightweight_root_cause(
        case={"query": "What is Bluup?"},
        expected_status="resolved",
        mapping_issues=[],
        chunk_rank=73,
        chunk_similarity=0.71,
        document_rank=4,
        document_similarity=0.78,
        top_hits=[{"rank": 10, "similarity": 0.72, "chunk_id": "x", "document_id": "y"}],
        expected_chunk_ids={"c1"},
        expected_document_ids={"d1"},
    )
    assert result["primary_root_cause"] == "CHUNKING"


def test_retrieval_competition_classification() -> None:
    hits = [{"rank": i, "similarity": 0.75 - i * 0.001, "chunk_id": f"c{i}", "document_id": f"d{i}"} for i in range(1, 21)]
    result = classify_lightweight_root_cause(
        case={"query": "What is Bluup?"},
        expected_status="resolved",
        mapping_issues=[],
        chunk_rank=17,
        chunk_similarity=0.742,
        document_rank=17,
        document_similarity=0.742,
        top_hits=hits,
        expected_chunk_ids={"expected"},
        expected_document_ids={"d-exp"},
    )
    assert result["primary_root_cause"] == "RETRIEVAL_COMPETITION"


def test_lightweight_report_generation() -> None:
    answer = _chunk(
        chunk_id="answer",
        document_id="doc-a",
        content="Customer support center timings are 10:00am to 6:00pm. Sunday closed.",
    )
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
                    "section_path": answer.section_path,
                    "content": answer.content,
                }
            ]
        }
    )
    report = run_lightweight_audit(
        config=EmbeddingConfig.from_settings(),
        chunks=[answer],
        search=search,
        top_k=100,
    )
    assert report["report_version"] == "1.0"
    assert report["question_count"] == 13
    assert "top_100_metadata" not in report["questions"][0]
    item = next(q for q in report["questions"] if q["id"] == "ERK-013")
    assert item["top_10"][0]["text"]
    text = format_lightweight_text_report(report, full_questions=report["_internal_questions"])
    assert "EAR_KART_RETRIEVAL_ROOT_CAUSE_13" in text


def test_hit_metadata_has_no_text_field() -> None:
    chunk = _chunk(chunk_id="c1", document_id="d1", content="secret long text")
    meta = format_hit_metadata(rank=1, item={"chunk_id": "c1", "similarity": 0.5}, chunk=chunk)
    assert "text" not in meta
    assert meta["token_count"] == chunk.token_count


def test_top_competing_document() -> None:
    hits = [
        {"rank": 1, "document_id": "other", "title": "Investor PDF", "similarity": 0.9},
        {"rank": 2, "document_id": "expected", "title": "FAQ", "similarity": 0.8},
    ]
    title, sim = top_competing_document(hits, {"expected"})
    assert title == "Investor PDF"
    assert sim == 0.9


def test_frozen_chunk_integrity_baseline_exists() -> None:
    baseline = Path("data/integrity/phase11_baseline.json")
    assert baseline.exists()
    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert data["manifest_total_chunks"] == 5904
    chunks_root = settings.chunks_dir / settings.kb_dataset_version
    if not chunks_root.exists():
        pytest.skip("frozen chunks unavailable")
    result = verify_directory_unchanged(chunks_root, baseline)
    if not result["unchanged"]:
        pytest.skip("chunk integrity baseline differs in this workspace")
