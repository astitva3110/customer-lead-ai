"""Shared retrieval case evaluation helpers."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.metrics import recall_at_k, reciprocal_rank
from app.kb.vector.search import VectorSearchService


def evaluate_retrieval_cases(
    cases: list[dict[str, Any]],
    search: VectorSearchService,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        query_results = search.search(case["query"], top_k=top_k)
        retrieved_ids = [item["chunk_id"] for item in query_results]
        expected_ids = case["expected_chunk_ids"]
        require_all = case.get("require_all_expected_chunks", False)
        results.append(
            {
                **case,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved": [
                    {
                        "chunk_id": item["chunk_id"],
                        "document_id": item["document_id"],
                        "similarity": item.get("similarity"),
                        "source_url": item.get("source_url"),
                    }
                    for item in query_results[:top_k]
                ],
                "recall_at_1": recall_at_k(expected_ids, retrieved_ids, 1, require_all=require_all),
                "recall_at_3": recall_at_k(expected_ids, retrieved_ids, 3, require_all=require_all),
                "recall_at_5": recall_at_k(expected_ids, retrieved_ids, 5, require_all=require_all),
                "recall_at_10": recall_at_k(expected_ids, retrieved_ids, 10, require_all=require_all),
                "mrr": reciprocal_rank(expected_ids, retrieved_ids, require_all=require_all),
                "passed": recall_at_k(expected_ids, retrieved_ids, top_k, require_all=require_all) == 1.0,
            }
        )
    return results
