"""Eval-only candidate fusion. Does not change production HybridRetriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.services.retrieval.merge import merge_candidates
from app.services.retrieval.models import RetrievalCandidate
from app.helpers.retrieval_pool import cap_merged_candidates

StrategyName = Literal["current_merge", "rrf", "weighted", "rank_normalized"]


@dataclass(frozen=True)
class FusionConfig:
    top_k: int = 10
    cap: int | None = 30
    rrf_k: int = 60
    weighted_vector: float = 0.7
    weighted_keyword: float = 0.3
    rank_vector: float = 0.5
    rank_keyword: float = 0.5

    @classmethod
    def from_dict(cls, data: dict | None) -> FusionConfig:
        payload = dict(data or {})
        return cls(
            top_k=int(payload.get("top_k", 10)),
            cap=payload.get("cap", 30),
            rrf_k=int(payload.get("rrf_k", 60)),
            weighted_vector=float(payload.get("weighted_vector", 0.7)),
            weighted_keyword=float(payload.get("weighted_keyword", 0.3)),
            rank_vector=float(payload.get("rank_vector", 0.5)),
            rank_keyword=float(payload.get("rank_keyword", 0.5)),
        )


def fuse(
    name: StrategyName,
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    config: FusionConfig,
) -> list[RetrievalCandidate]:
    if name == "current_merge":
        return current_merge(vector, keyword, config)
    if name == "rrf":
        return rrf_fusion(vector, keyword, config)
    if name == "weighted":
        return weighted_fusion(vector, keyword, config)
    if name == "rank_normalized":
        return rank_normalized_fusion(vector, keyword, config)
    raise ValueError(f"unknown fusion strategy: {name}")


def current_merge(
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    config: FusionConfig,
) -> list[RetrievalCandidate]:
    """Production union + cap. Vector-first unless cap re-sorts by max score."""
    merged = merge_candidates(vector, keyword)
    capped = cap_merged_candidates(merged, config.cap)
    return capped[: config.top_k]


def rrf_fusion(
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    config: FusionConfig,
) -> list[RetrievalCandidate]:
    scores: dict[str, float] = {}
    for rank, candidate in enumerate(vector, start=1):
        scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (config.rrf_k + rank)
    for rank, candidate in enumerate(keyword, start=1):
        scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (config.rrf_k + rank)
    return _order_union(vector, keyword, scores, config.top_k)


def weighted_fusion(
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    config: FusionConfig,
) -> list[RetrievalCandidate]:
    vector_norm = _minmax({item.chunk_id: item.vector_score or 0.0 for item in vector})
    keyword_norm = _minmax({item.chunk_id: item.keyword_score or 0.0 for item in keyword})
    ids = set(vector_norm) | set(keyword_norm)
    scores = {
        chunk_id: config.weighted_vector * vector_norm.get(chunk_id, 0.0)
        + config.weighted_keyword * keyword_norm.get(chunk_id, 0.0)
        for chunk_id in ids
    }
    return _order_union(vector, keyword, scores, config.top_k)


def rank_normalized_fusion(
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    config: FusionConfig,
) -> list[RetrievalCandidate]:
    vector_rank = _rank_scores(vector)
    keyword_rank = _rank_scores(keyword)
    ids = set(vector_rank) | set(keyword_rank)
    scores = {
        chunk_id: config.rank_vector * vector_rank.get(chunk_id, 0.0)
        + config.rank_keyword * keyword_rank.get(chunk_id, 0.0)
        for chunk_id in ids
    }
    return _order_union(vector, keyword, scores, config.top_k)


def _order_union(
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    scores: dict[str, float],
    top_k: int,
) -> list[RetrievalCandidate]:
    combined = {item.chunk_id: item for item in merge_candidates(vector, keyword)}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [combined[chunk_id] for chunk_id, _score in ordered if chunk_id in combined][:top_k]


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _rank_scores(rows: list[RetrievalCandidate]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}
    return {item.chunk_id: (n - index) / n for index, item in enumerate(rows)}
