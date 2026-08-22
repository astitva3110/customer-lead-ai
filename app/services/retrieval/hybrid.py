from __future__ import annotations

import time

from app.interfaces.providers.candidate_retriever import CandidateRetriever
from app.interfaces.providers.reranker import Reranker
from app.services.retrieval.merge import merge_candidates
from app.services.retrieval.models import HybridSearchResult, StageTimings
from app.services.retrieval.threshold import apply_rerank_threshold, take_final_k
from app.helpers.retrieval_pool import cap_merged_candidates


class HybridRetriever:
    """Orchestrates vector + keyword candidate generation, merge, rerank, and threshold."""

    def __init__(
        self,
        vector_retriever: CandidateRetriever,
        keyword_retriever: CandidateRetriever,
        reranker: Reranker,
        *,
        vector_k: int = 20,
        keyword_k: int = 20,
        final_k: int = 5,
        min_score: float = 0.0,
        rerank_candidate_k: int | None = None,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._reranker = reranker
        self._vector_k = vector_k
        self._keyword_k = keyword_k
        self._final_k = final_k
        self._min_score = min_score
        self._rerank_candidate_k = rerank_candidate_k

    def search_detailed(
        self,
        query: str,
        *,
        final_k: int | None = None,
        document_id: str | None = None,
        min_score: float | None = None,
    ) -> HybridSearchResult:
        started = time.perf_counter()
        timings = StageTimings()
        threshold = self._min_score if min_score is None else min_score
        top_k = self._final_k if final_k is None else final_k

        vector_started = time.perf_counter()
        vector_candidates = self._vector_retriever.retrieve(
            query,
            top_k=self._vector_k,
            document_id=document_id,
        )
        timings.vector_ms = (time.perf_counter() - vector_started) * 1000

        keyword_started = time.perf_counter()
        keyword_candidates = self._keyword_retriever.retrieve(
            query,
            top_k=self._keyword_k,
            document_id=document_id,
        )
        timings.keyword_ms = (time.perf_counter() - keyword_started) * 1000

        merge_started = time.perf_counter()
        merged = cap_merged_candidates(
            merge_candidates(vector_candidates, keyword_candidates),
            self._rerank_candidate_k,
        )
        timings.merge_ms = (time.perf_counter() - merge_started) * 1000

        rerank_started = time.perf_counter()
        reranked = self._reranker.rerank(query, merged)
        timings.rerank_ms = (time.perf_counter() - rerank_started) * 1000

        threshold_started = time.perf_counter()
        surviving = apply_rerank_threshold(reranked, threshold)
        final = take_final_k(surviving, top_k)
        timings.threshold_ms = (time.perf_counter() - threshold_started) * 1000
        timings.total_ms = (time.perf_counter() - started) * 1000

        return HybridSearchResult(
            query=query,
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            merged_candidates=merged,
            reranked_candidates=reranked,
            final_candidates=final,
            min_score=threshold,
            has_relevant_context=bool(final),
            timings=timings,
        )

    def search(self, query: str, *, top_k: int | None = None, document_id: str | None = None):
        """Retriever-protocol search: final ranked hits only."""
        return self.search_detailed(query, final_k=top_k, document_id=document_id).final_candidates
