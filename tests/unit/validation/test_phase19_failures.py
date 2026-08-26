from __future__ import annotations

import pytest

from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


class _OkRetriever:
    def retrieve(self, query: str, *, top_k: int, document_id: str | None = None):
        del query, document_id
        return [
            RetrievalCandidate(chunk_id="c1", document_id="d1", text="TINY is a CIC.", vector_score=0.9)
        ][:top_k]


class _BoomRetriever:
    def retrieve(self, query: str, *, top_k: int, document_id: str | None = None):
        raise RuntimeError("keyword backend down")


class _OkReranker:
    def rerank(self, query: str, candidates: list[RetrievalCandidate]):
        del query
        return candidates


class _BoomReranker:
    def rerank(self, query: str, candidates: list[RetrievalCandidate]):
        raise RuntimeError("reranker down")


def test_keyword_failure_is_not_swallowed() -> None:
    hybrid = HybridRetriever(_OkRetriever(), _BoomRetriever(), _OkReranker(), min_score=0.0)
    with pytest.raises(RuntimeError, match="keyword backend down"):
        hybrid.search_detailed("TINY")


def test_reranker_failure_is_not_swallowed() -> None:
    hybrid = HybridRetriever(_OkRetriever(), _OkRetriever(), _BoomReranker(), min_score=0.0)
    with pytest.raises(RuntimeError, match="reranker down"):
        hybrid.search_detailed("TINY")


def test_vector_only_fallback_not_implemented() -> None:
    hybrid = HybridRetriever(_OkRetriever(), _BoomRetriever(), _OkReranker(), min_score=0.0)
    with pytest.raises(RuntimeError):
        hybrid.search_detailed("TINY")


def test_llm_timeout_fail_closed() -> None:
    from tests.validation.harness import RecordingLiteLLM

    orchestrator, recorder, *_ = make_graph_stack(
        knowledge=FakeKnowledge(),
        recorder=RecordingLiteLLM("timeout"),
    )
    result = orchestrator.handle("fail-llm", "What is TINY?")
    assert result.mode == "KNOWLEDGE"
    assert "enough information" in result.response.lower()
    assert recorder.call_count >= 1
    assert result.sources == []
