"""Combined retrieval evaluation: qc35 + golden Excel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.case_runner import evaluate_retrieval_cases
from app.kb.evaluation.dataset import EVALUATION_DATASET_VERSION, load_or_build_dataset, validate_dataset
from app.kb.evaluation.golden_excel import (
    DEFAULT_GOLDEN_EXCEL_PATH,
    DEFAULT_GOLDEN_SHEET,
    GOLDEN_DATASET_VERSION,
    enrich_golden_cases,
    load_golden_cases_from_excel,
    validate_golden_cases,
)
from app.kb.evaluation.metrics import aggregate_metrics
from app.kb.vector.search import VectorSearchService

COMBINED_DATASET_VERSION = "combined_qc35_golden_v1"


def _normalize_qc35_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "dataset": EVALUATION_DATASET_VERSION,
        "query": case["query"],
        "category": case.get("category", "general"),
        "importance": case.get("importance", "standard"),
        "expected_document_id": case.get("expected_document_id"),
        "expected_chunk_ids": case["expected_chunk_ids"],
        "require_all_expected_chunks": case.get("require_all_expected_chunks", False),
        "expected_needles": case.get("expected_needles", []),
    }


def _normalize_golden_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "dataset": GOLDEN_DATASET_VERSION,
        "query": case["query"],
        "category": case.get("category", "general"),
        "importance": "standard",
        "expected_document_id": case.get("expected_document_id"),
        "expected_chunk_ids": case["expected_chunk_ids"],
        "require_all_expected_chunks": case.get("require_all_expected_chunks", False),
        "expected_answer": case.get("expected_answer", ""),
        "grounding_points": case.get("grounding_points", []),
        "expected_source": case.get("expected_source"),
        "source_row": case.get("source_row"),
        "source_sheet": case.get("source_sheet"),
    }


class CombinedRetrievalEvaluationRunner:
    """Run chunk-ID recall on qc35 and golden Excel questions together."""

    def __init__(
        self,
        *,
        config: EmbeddingConfig,
        loader: ChunkLoader,
        search: VectorSearchService,
        excel_path: Path = DEFAULT_GOLDEN_EXCEL_PATH,
        sheet_name: str | None = DEFAULT_GOLDEN_SHEET,
    ) -> None:
        self.config = config
        self.loader = loader
        self.search = search
        self.excel_path = excel_path
        self.sheet_name = sheet_name

    def run(self, *, top_k: int = 10) -> dict[str, Any]:
        chunks = self.loader.load_all().chunks

        qc35_dataset = load_or_build_dataset(chunks)
        validate_dataset(qc35_dataset)

        golden_raw = load_golden_cases_from_excel(self.excel_path, sheet_name=self.sheet_name)
        golden_dataset, golden_skipped = enrich_golden_cases(golden_raw, chunks)
        validate_golden_cases(golden_dataset)

        cases = [_normalize_qc35_case(case) for case in qc35_dataset]
        cases.extend(_normalize_golden_case(case) for case in golden_dataset)

        results = evaluate_retrieval_cases(cases, self.search, top_k=top_k)
        qc35_results = [item for item in results if item["dataset"] == EVALUATION_DATASET_VERSION]
        golden_results = [item for item in results if item["dataset"] == GOLDEN_DATASET_VERSION]

        combined_metrics = aggregate_metrics(results)
        qc35_metrics = aggregate_metrics(qc35_results)
        golden_metrics = aggregate_metrics(golden_results)

        return {
            "evaluation_dataset_version": COMBINED_DATASET_VERSION,
            "datasets": {
                EVALUATION_DATASET_VERSION: {
                    "source": "app/kb/chunking/question_coverage.py",
                    "total_queries": qc35_metrics["total"],
                    "passed_queries": qc35_metrics["passed"],
                    "skipped_queries": 0,
                    "metrics": qc35_metrics,
                },
                GOLDEN_DATASET_VERSION: {
                    "source": str(self.excel_path),
                    "sheet": self.sheet_name or DEFAULT_GOLDEN_SHEET,
                    "total_rows": len(golden_raw),
                    "total_queries": golden_metrics["total"],
                    "passed_queries": golden_metrics["passed"],
                    "skipped_queries": len(golden_skipped),
                    "skipped_cases": golden_skipped,
                    "metrics": golden_metrics,
                },
            },
            "embedding_version": self.config.embedding_version,
            "embedding_model": self.config.model,
            "embedding_model_revision": self.config.model_revision,
            "embedding_dimension": self.config.dimension,
            "kb_dataset_version": self.config.kb_dataset_version,
            "chunking_algorithm_version": self.config.chunking_algorithm_version,
            "embedding_input_manifest": self.config.embedding_input_manifest,
            "metrics": combined_metrics,
            "total_evaluation_queries": combined_metrics["total"],
            "passed_queries": combined_metrics["passed"],
            "failed_queries": combined_metrics["total"] - combined_metrics["passed"],
            "results": results,
            "passed": combined_metrics["passed"] == combined_metrics["total"],
        }
