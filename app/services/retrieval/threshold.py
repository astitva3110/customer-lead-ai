from __future__ import annotations

from app.services.retrieval.models import RankedCandidate


def apply_rerank_threshold(
    candidates: list[RankedCandidate],
    min_score: float,
) -> list[RankedCandidate]:
    """Keep candidates whose rerank_score meets the configured minimum."""
    kept: list[RankedCandidate] = []
    for candidate in candidates:
        score = candidate.rerank_score
        if score is None:
            continue
        if score >= min_score:
            kept.append(candidate)
    return kept


def take_final_k(candidates: list[RankedCandidate], final_k: int) -> list[RankedCandidate]:
    if final_k < 0:
        raise ValueError("final_k must be >= 0")
    return candidates[:final_k]
