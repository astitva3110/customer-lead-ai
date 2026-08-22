from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.helpers.retrieval_pool import cap_merged_candidates
from app.providers.reranker.passthrough import PassthroughReranker


def _candidate(chunk_id: str, *, vector: float | None = None, keyword: float | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=chunk_id,
        vector_score=vector,
        keyword_score=keyword,
    )


def test_cap_merged_candidates_keeps_highest_scores() -> None:
    merged = [_candidate(str(index), vector=1.0 - (index * 0.01)) for index in range(40)]
    capped = cap_merged_candidates(merged, 30)
    assert len(capped) == 30
    assert capped[0].chunk_id == "0"
    assert cap_merged_candidates(merged, None) == merged


def test_hybrid_rerank_candidate_k_caps_before_rerank() -> None:
    class _Retriever:
        def __init__(self, items: list[RetrievalCandidate]) -> None:
            self.items = items

        def retrieve(self, query: str, *, top_k: int, document_id: str | None = None):
            del query, document_id
            return self.items[:top_k]

    class _RecordingReranker:
        def __init__(self) -> None:
            self.seen_ids: list[str] = []
            self._inner = PassthroughReranker()

        def rerank(self, query: str, candidates: list[RetrievalCandidate]):
            self.seen_ids = [item.chunk_id for item in candidates]
            return self._inner.rerank(query, candidates)

    reranker = _RecordingReranker()
    hybrid = HybridRetriever(
        _Retriever([_candidate(f"v{i}", vector=0.9 - i * 0.01) for i in range(20)]),
        _Retriever([_candidate(f"k{i}", keyword=0.8 - i * 0.01) for i in range(20)]),
        reranker,
        vector_k=20,
        keyword_k=20,
        final_k=6,
        rerank_candidate_k=30,
    )
    result = hybrid.search_detailed("tiny")
    assert len(reranker.seen_ids) == 30
    assert len(result.merged_candidates) == 30
    assert len(result.final_candidates) == 6
