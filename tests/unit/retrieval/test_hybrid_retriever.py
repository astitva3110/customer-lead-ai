from dataclasses import replace

from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.services.retrieval.diagnostics import format_retrieval_trace


class _FakeRetriever:
    def __init__(self, items: list[RetrievalCandidate]) -> None:
        self.items = items
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, *, top_k: int, document_id: str | None = None):
        self.calls.append((query, top_k, document_id))
        return self.items[:top_k]


class _FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen_ids: list[str] | None = None

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        del query
        self.seen_ids = [item.chunk_id for item in candidates]
        ranked = [replace(item, rerank_score=self.scores[item.chunk_id]) for item in candidates]
        ranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return ranked


def _candidate(chunk_id: str, *, vector: float | None = None, keyword: float | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=chunk_id,
        vector_score=vector,
        keyword_score=keyword,
    )


def test_hybrid_retriever_flow_merge_rerank_threshold_final_k() -> None:
    vector = _FakeRetriever(
        [
            _candidate("A", vector=0.9),
            _candidate("B", vector=0.8),
            _candidate("C", vector=0.7),
        ]
    )
    keyword = _FakeRetriever(
        [
            _candidate("B", keyword=0.6),
            _candidate("C", keyword=0.5),
            _candidate("D", keyword=0.4),
        ]
    )
    reranker = _FakeReranker({"A": 0.91, "B": 0.82, "C": 0.40, "D": 0.88})
    hybrid = HybridRetriever(
        vector,
        keyword,
        reranker,
        vector_k=20,
        keyword_k=20,
        final_k=2,
        min_score=0.80,
    )
    result = hybrid.search_detailed("radius 16")
    assert vector.calls == [("radius 16", 20, None)]
    assert keyword.calls == [("radius 16", 20, None)]
    assert reranker.seen_ids == ["A", "B", "C", "D"]
    assert [item.chunk_id for item in result.merged_candidates] == ["A", "B", "C", "D"]
    assert result.merged_candidates[1].vector_score == 0.8
    assert result.merged_candidates[1].keyword_score == 0.6
    assert [item.chunk_id for item in result.reranked_candidates] == ["A", "D", "B", "C"]
    assert [item.chunk_id for item in result.final_candidates] == ["A", "D"]
    assert result.has_relevant_context is True
    hits = hybrid.search("radius 16")
    assert [item.chunk_id for item in hits] == ["A", "D"]


def test_hybrid_no_result_when_all_below_threshold() -> None:
    vector = _FakeRetriever([_candidate("A", vector=0.5), _candidate("B", vector=0.4)])
    keyword = _FakeRetriever([])
    reranker = _FakeReranker({"A": 0.50, "B": 0.40})
    hybrid = HybridRetriever(vector, keyword, reranker, min_score=0.80, final_k=5)
    result = hybrid.search_detailed("unknown topic")
    assert result.has_relevant_context is False
    assert result.results == []
    assert hybrid.search("unknown topic") == []
    trace = format_retrieval_trace(result)
    assert "THRESHOLD:" in trace
    assert "0.8" in trace
    assert "has_relevant_context=False" in trace
