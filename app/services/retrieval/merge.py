from dataclasses import replace

from app.services.retrieval.models import RetrievalCandidate


def merge_candidates(
    vector_candidates: list[RetrievalCandidate],
    keyword_candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Union vector and keyword hits by chunk_id, preserving both scores."""
    merged: dict[str, RetrievalCandidate] = {}
    order: list[str] = []

    for candidate in vector_candidates:
        if candidate.chunk_id not in merged:
            order.append(candidate.chunk_id)
            merged[candidate.chunk_id] = candidate
            continue
        existing = merged[candidate.chunk_id]
        merged[candidate.chunk_id] = replace(
            candidate,
            keyword_score=existing.keyword_score if existing.keyword_score is not None else candidate.keyword_score,
        )

    for candidate in keyword_candidates:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            order.append(candidate.chunk_id)
            merged[candidate.chunk_id] = candidate
            continue
        merged[candidate.chunk_id] = replace(
            existing,
            keyword_score=candidate.keyword_score,
            vector_score=existing.vector_score if existing.vector_score is not None else candidate.vector_score,
        )

    return [merged[chunk_id] for chunk_id in order]
