from __future__ import annotations

from dataclasses import replace

from app.services.retrieval.models import RankedCandidate, RetrievalCandidate


class PassthroughReranker:
    """Keeps retrieval order unless scores are already set; uses vector then keyword."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RankedCandidate]:
        del query
        scored: list[RankedCandidate] = []
        for candidate in candidates:
            score = candidate.rerank_score
            if score is None:
                score = candidate.vector_score
            if score is None:
                score = candidate.keyword_score
            scored.append(replace(candidate, rerank_score=float(score or 0.0)))
        scored.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return scored
