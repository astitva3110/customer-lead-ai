"""Retrieval evaluation metrics."""

from __future__ import annotations


def recall_at_k(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int,
    *,
    require_all: bool = False,
) -> float:
    if not expected_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    if require_all:
        return 1.0 if all(chunk_id in top_k for chunk_id in expected_chunk_ids) else 0.0
    return 1.0 if any(chunk_id in top_k for chunk_id in expected_chunk_ids) else 0.0


def reciprocal_rank(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    *,
    require_all: bool = False,
) -> float:
    if not expected_chunk_ids:
        return 0.0
    if require_all and len(expected_chunk_ids) > 1:
        ranks: list[int] = []
        for chunk_id in expected_chunk_ids:
            for index, retrieved_id in enumerate(retrieved_chunk_ids, start=1):
                if retrieved_id == chunk_id:
                    ranks.append(index)
                    break
            else:
                return 0.0
        return 1.0 / max(ranks)
    expected = set(expected_chunk_ids)
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / index
    return 0.0


def aggregate_metrics(results: list[dict]) -> dict:
    if not results:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "total": 0,
            "passed": 0,
        }
    total = len(results)
    return {
        "recall_at_1": round(sum(r["recall_at_1"] for r in results) / total, 4),
        "recall_at_3": round(sum(r["recall_at_3"] for r in results) / total, 4),
        "recall_at_5": round(sum(r["recall_at_5"] for r in results) / total, 4),
        "recall_at_10": round(sum(r["recall_at_10"] for r in results) / total, 4),
        "mrr": round(sum(r["mrr"] for r in results) / total, 4),
        "total": total,
        "passed": sum(1 for r in results if r["passed"]),
    }
