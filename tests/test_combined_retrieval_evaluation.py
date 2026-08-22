"""Tests for combined retrieval evaluation."""

from __future__ import annotations

from app.kb.evaluation.case_runner import evaluate_retrieval_cases
from app.kb.evaluation.combined_runner import (
    COMBINED_DATASET_VERSION,
    CombinedRetrievalEvaluationRunner,
    _normalize_golden_case,
    _normalize_qc35_case,
)
from app.kb.evaluation.metrics import aggregate_metrics


class _FakeSearch:
    def search(self, query: str, *, top_k: int = 10):
        if "returns" in query.lower():
            return [
                {
                    "chunk_id": "chunk-a",
                    "document_id": "doc-a",
                    "similarity": 0.9,
                    "source_url": "https://example.com/returns",
                }
            ]
        return [
            {
                "chunk_id": "chunk-b",
                "document_id": "doc-b",
                "similarity": 0.5,
                "source_url": "https://example.com/other",
            }
        ]


def test_evaluate_retrieval_cases_recall() -> None:
    cases = [
        {
            "id": "T1",
            "dataset": "test",
            "query": "Where do I ship returns?",
            "expected_chunk_ids": ["chunk-a"],
            "require_all_expected_chunks": False,
        }
    ]
    results = evaluate_retrieval_cases(cases, _FakeSearch(), top_k=5)
    assert results[0]["recall_at_1"] == 1.0
    assert results[0]["passed"] is True


def test_normalize_case_shapes() -> None:
    qc = _normalize_qc35_case(
        {
            "id": "QC01",
            "query": "test?",
            "category": "return_policy",
            "expected_chunk_ids": ["abc"],
            "require_all_expected_chunks": False,
            "expected_needles": ["needle"],
        }
    )
    golden = _normalize_golden_case(
        {
            "id": "C01-01",
            "query": "test?",
            "category": "Product",
            "expected_chunk_ids": ["abc"],
            "require_all_expected_chunks": False,
            "expected_answer": "answer",
            "grounding_points": ["fact"],
        }
    )
    assert qc["dataset"] == "qc35_v1"
    assert golden["dataset"] == "golden_excel_v1"
    assert COMBINED_DATASET_VERSION == "combined_qc35_golden_v1"


def test_aggregate_combined_metrics() -> None:
    results = [
        {"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "passed": True},
        {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 0.25, "passed": True},
    ]
    metrics = aggregate_metrics(results)
    assert metrics["recall_at_1"] == 0.5
    assert metrics["passed"] == 2
