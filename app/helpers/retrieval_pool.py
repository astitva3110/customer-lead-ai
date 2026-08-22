from __future__ import annotations

from app.services.retrieval.models import RetrievalCandidate


def cap_merged_candidates(
    merged: list[RetrievalCandidate],
    limit: int | None,
) -> list[RetrievalCandidate]:
    if limit is None or len(merged) <= limit:
        return merged

    def score(candidate: RetrievalCandidate) -> float:
        return max(candidate.vector_score or 0.0, candidate.keyword_score or 0.0)

    return sorted(merged, key=score, reverse=True)[:limit]
