"""Chunk-ID retrieval evaluation for Excel golden Q&A datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.golden_excel import (
    DEFAULT_GOLDEN_SHEET,
    GOLDEN_DATASET_VERSION,
    enrich_golden_cases,
    load_golden_cases_from_excel,
    validate_golden_cases,
)
from app.kb.evaluation.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from app.kb.vector.search import VectorSearchService


class GoldenExcelEvaluationRunner:
    """Same chunk-ID recall flow as phase11 RetrievalEvaluationRunner."""

    def __init__(
        self,
        *,
        config: EmbeddingConfig,
        loader: ChunkLoader,
        search: VectorSearchService,
        excel_path: Path,
        sheet_name: str | None = None,
    ) -> None:
        self.config = config
        self.loader = loader
        self.search = search
        self.excel_path = excel_path
        self.sheet_name = sheet_name

    def run(self, *, top_k: int = 10, resolve_only: bool = False) -> dict[str, Any]:
        chunks = self.loader.load_all().chunks
        raw_cases = load_golden_cases_from_excel(self.excel_path, sheet_name=self.sheet_name)
        dataset, skipped_cases = enrich_golden_cases(raw_cases, chunks)

        if resolve_only:
            return {
                "evaluation_dataset_version": GOLDEN_DATASET_VERSION,
                "excel_path": str(self.excel_path),
                "sheet_name": self.sheet_name or DEFAULT_GOLDEN_SHEET,
                "total_rows": len(raw_cases),
                "resolved_cases": len(dataset),
                "skipped_cases": len(skipped_cases),
                "resolved": dataset,
                "skipped": skipped_cases,
                "resolve_only": True,
                "passed": True,
            }

        validate_golden_cases(dataset)

        per_query: list[dict[str, Any]] = []
        for case in dataset:
            results = self.search.search(case["query"], top_k=top_k)
            retrieved_ids = [item["chunk_id"] for item in results]
            expected_ids = case["expected_chunk_ids"]
            require_all = case.get("require_all_expected_chunks", False)
            per_query.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "category": case.get("category", "general"),
                    "expected_answer": case.get("expected_answer", ""),
                    "grounding_points": case.get("grounding_points", []),
                    "expected_source": case.get("expected_source"),
                    "expected_chunk_ids": expected_ids,
                    "require_all_expected_chunks": require_all,
                    "source_row": case.get("source_row"),
                    "source_sheet": case.get("source_sheet"),
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved": [
                        {
                            "chunk_id": result["chunk_id"],
                            "document_id": result["document_id"],
                            "similarity": result.get("similarity"),
                            "source_url": result.get("source_url"),
                        }
                        for result in results[:top_k]
                    ],
                    "recall_at_1": recall_at_k(expected_ids, retrieved_ids, 1, require_all=require_all),
                    "recall_at_3": recall_at_k(expected_ids, retrieved_ids, 3, require_all=require_all),
                    "recall_at_5": recall_at_k(expected_ids, retrieved_ids, 5, require_all=require_all),
                    "recall_at_10": recall_at_k(expected_ids, retrieved_ids, 10, require_all=require_all),
                    "mrr": reciprocal_rank(expected_ids, retrieved_ids, require_all=require_all),
                    "passed": recall_at_k(expected_ids, retrieved_ids, top_k, require_all=require_all) == 1.0,
                }
            )

        metrics = aggregate_metrics(per_query)
        return {
            "evaluation_dataset_version": GOLDEN_DATASET_VERSION,
            "excel_path": str(self.excel_path),
            "sheet_name": self.sheet_name or DEFAULT_GOLDEN_SHEET,
            "embedding_version": self.config.embedding_version,
            "embedding_model": self.config.model,
            "embedding_model_revision": self.config.model_revision,
            "embedding_dimension": self.config.dimension,
            "kb_dataset_version": self.config.kb_dataset_version,
            "chunking_algorithm_version": self.config.chunking_algorithm_version,
            "embedding_input_manifest": self.config.embedding_input_manifest,
            "metrics": metrics,
            "total_rows": len(raw_cases),
            "skipped_rows": len(skipped_cases),
            "skipped_cases": skipped_cases,
            "total_evaluation_queries": metrics["total"],
            "passed_queries": metrics["passed"],
            "failed_queries": metrics["total"] - metrics["passed"],
            "results": per_query,
            "passed": metrics["passed"] == metrics["total"],
        }
