"""Retrieval evaluation runner."""

from __future__ import annotations

from typing import Any

from app.kb.embedding.config import EmbeddingConfig
from app.kb.evaluation.dataset import EVALUATION_DATASET_VERSION, load_or_build_dataset, validate_dataset
from app.kb.evaluation.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from app.kb.embedding.loader import ChunkLoader
from app.kb.vector.search import VectorSearchService


class RetrievalEvaluationRunner:
    def __init__(
        self,
        *,
        config: EmbeddingConfig,
        loader: ChunkLoader,
        search: VectorSearchService,
    ) -> None:
        self.config = config
        self.loader = loader
        self.search = search

    def run(self, *, top_k: int = 10) -> dict[str, Any]:
        chunks = self.loader.load_all().chunks
        dataset = load_or_build_dataset(chunks)
        validate_dataset(dataset)

        per_query: list[dict[str, Any]] = []
        for case in dataset:
            results = self.search.search(case["query"], top_k=top_k)
            retrieved_ids = [item["chunk_id"] for item in results]
            expected_ids = case["expected_chunk_ids"]
            require_all = case.get("require_all_expected_chunks", False)
            item = {
                "id": case["id"],
                "query": case["query"],
                "category": case["category"],
                "importance": case["importance"],
                "expected_document_id": case.get("expected_document_id"),
                "expected_chunk_ids": expected_ids,
                "require_all_expected_chunks": require_all,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved": [
                    {
                        "chunk_id": r["chunk_id"],
                        "document_id": r["document_id"],
                        "similarity": r.get("similarity"),
                        "source_url": r.get("source_url"),
                    }
                    for r in results[:top_k]
                ],
                "recall_at_1": recall_at_k(expected_ids, retrieved_ids, 1, require_all=require_all),
                "recall_at_3": recall_at_k(expected_ids, retrieved_ids, 3, require_all=require_all),
                "recall_at_5": recall_at_k(expected_ids, retrieved_ids, 5, require_all=require_all),
                "recall_at_10": recall_at_k(expected_ids, retrieved_ids, 10, require_all=require_all),
                "mrr": reciprocal_rank(expected_ids, retrieved_ids, require_all=require_all),
                "passed": recall_at_k(expected_ids, retrieved_ids, top_k, require_all=require_all) == 1.0,
            }
            per_query.append(item)

        metrics = aggregate_metrics(per_query)
        return {
            "evaluation_dataset_version": EVALUATION_DATASET_VERSION,
            "embedding_version": self.config.embedding_version,
            "embedding_model": self.config.model,
            "embedding_model_revision": self.config.model_revision,
            "embedding_dimension": self.config.dimension,
            "kb_dataset_version": self.config.kb_dataset_version,
            "chunking_algorithm_version": self.config.chunking_algorithm_version,
            "embedding_input_manifest": self.config.embedding_input_manifest,
            "metrics": metrics,
            "total_evaluation_queries": metrics["total"],
            "passed_queries": metrics["passed"],
            "failed_queries": metrics["total"] - metrics["passed"],
            "results": per_query,
            "passed": metrics["passed"] == metrics["total"],
        }
