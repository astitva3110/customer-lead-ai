"""Phase 15 KB V2 read-only retrieval evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.evaluation.metrics import recall_at_k, reciprocal_rank
from app.kb.evaluation.phase15_kb_v2_dataset import DATASET_NAME, build_phase15_cases
from app.kb.evaluation.phase15_root_cause import classify_phase15_root_cause
from app.kb.evaluation.retrieval_diagnostic.ranking import (
    summarize_expected_chunk_analysis,
    summarize_expected_document_analysis,
)
from app.kb.ingestion.indexing import Phase12VectorStore


@dataclass
class Phase15KbV2Search:
    store: Phase12VectorStore
    provider: Any

    @classmethod
    def from_settings(cls, *, device: str | None = None) -> Phase15KbV2Search:
        if device:
            import os

            os.environ["EMBEDDING_DEVICE"] = device
        from dataclasses import replace

        config = replace(EmbeddingConfig.from_settings(), device=device or settings.embedding_device)
        provider = create_embedding_provider(config)
        store = Phase12VectorStore(table_name=settings.phase12_vector_table)
        return cls(store=store, provider=provider)

    def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        vector = self.provider.embed_queries([query])[0]
        return self.store.search(
            vector,
            top_k=top_k,
            embedding_version=settings.phase12_embedding_version,
        )


def _format_hit(rank: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "similarity": item.get("similarity"),
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "document_title": item.get("title"),
        "page_number": item.get("page_number"),
        "section_path": item.get("section_path") or [],
        "content_type": item.get("split_method") or item.get("source_type"),
        "token_count": _estimate_token_count(item.get("content") or ""),
        "text": item.get("content") or "",
    }


def _estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def _rank_hits(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_format_hit(index, item) for index, item in enumerate(raw_hits, start=1)]


def _aggregate_passed(results: list[dict[str, Any]]) -> dict[str, int]:
    answerable = [item for item in results if item.get("answerability") != "CORPUS_GAP"]
    return {
        "passed_at_1": sum(1 for item in answerable if item["recall_at_1"] == 1.0),
        "passed_at_3": sum(1 for item in answerable if item["recall_at_3"] == 1.0),
        "passed_at_5": sum(1 for item in answerable if item["recall_at_5"] == 1.0),
        "passed_at_10": sum(1 for item in answerable if item["recall_at_10"] == 1.0),
    }


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in results if item.get("answerability") != "CORPUS_GAP"]
    if not answerable:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
        }
    total = len(answerable)
    return {
        "recall_at_1": round(sum(item["recall_at_1"] for item in answerable) / total, 4),
        "recall_at_3": round(sum(item["recall_at_3"] for item in answerable) / total, 4),
        "recall_at_5": round(sum(item["recall_at_5"] for item in answerable) / total, 4),
        "recall_at_10": round(sum(item["recall_at_10"] for item in answerable) / total, 4),
        "mrr": round(sum(item["mrr"] for item in answerable) / total, 4),
    }


def run_phase15_kb_v2_evaluation(*, device: str | None = None) -> dict[str, Any]:
    cases = build_phase15_cases()
    search = Phase15KbV2Search.from_settings(device=device)
    query_results: list[dict[str, Any]] = []

    for case in cases:
        raw_100 = search.search(case["question"], top_k=100)
        ranked_100 = _rank_hits(raw_100)
        retrieved_ids = [item["chunk_id"] for item in ranked_100]
        expected_ids = case["expected_chunk_ids"]

        recall1 = recall_at_k(expected_ids, retrieved_ids, 1) if expected_ids else 0.0
        recall3 = recall_at_k(expected_ids, retrieved_ids, 3) if expected_ids else 0.0
        recall5 = recall_at_k(expected_ids, retrieved_ids, 5) if expected_ids else 0.0
        recall10 = recall_at_k(expected_ids, retrieved_ids, 10) if expected_ids else 0.0
        mrr = reciprocal_rank(expected_ids, retrieved_ids) if expected_ids else 0.0

        chunk_analysis = summarize_expected_chunk_analysis(ranked_100, expected_ids)
        document_analysis = summarize_expected_document_analysis(
            ranked_100,
            case["expected_document_ids"],
        )
        expected_chunk_rank = chunk_analysis.get("best_expected_chunk_rank")
        expected_document_best_rank = document_analysis.get("expected_document_best_rank")
        passed_at_10 = recall10 == 1.0

        root_cause = classify_phase15_root_cause(
            case=case,
            passed_at_10=passed_at_10,
            expected_chunk_rank=expected_chunk_rank,
            expected_document_best_rank=expected_document_best_rank,
            top_100_hits=ranked_100,
        )
        if passed_at_10 and root_cause is None:
            outcome = "EXPECTED_CHUNK_RETRIEVED"
        elif case["answerability"] == "CORPUS_GAP":
            outcome = "CORPUS_GAP"
        else:
            outcome = "FAILED"

        query_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answerability": case["answerability"],
                "expected_chunk_ids": expected_ids,
                "expected_document_ids": case["expected_document_ids"],
                "expected_section_path": case["expected_section_path"],
                "mapping_notes": case.get("mapping_notes", ""),
                "expected_chunk_rank": expected_chunk_rank,
                "expected_document_best_rank": expected_document_best_rank,
                "recall_at_1": recall1,
                "recall_at_3": recall3,
                "recall_at_5": recall5,
                "recall_at_10": recall10,
                "mrr": mrr,
                "passed_at_10": passed_at_10,
                "outcome": outcome,
                "root_cause": root_cause,
                "top_10_results": ranked_100[:10],
                "top_100_analysis": {
                    "expected_chunk_ranks": chunk_analysis.get("expected_chunk_ranks"),
                    "expected_document_in_top_10": document_analysis.get("expected_document_in_top_10"),
                    "expected_document_in_top_100": document_analysis.get("expected_document_in_top_100"),
                },
            }
        )

    metrics = _aggregate_metrics(query_results)
    passed = _aggregate_passed(query_results)
    root_cause_summary = _summarize_root_causes(query_results)

    return {
        "dataset": DATASET_NAME,
        "vector_table": settings.phase12_vector_table,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_version": settings.phase12_embedding_version,
        "read_only": True,
        "total_questions": len(cases),
        "metrics": metrics,
        "passed": passed,
        "summary": _build_outcome_summary(query_results),
        "root_cause_summary": root_cause_summary,
        "queries": query_results,
    }


def _build_outcome_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    corpus_gaps = sum(1 for item in results if item["answerability"] == "CORPUS_GAP")
    expected_chunk_retrieved = sum(1 for item in results if item["passed_at_10"])
    expected_document_retrieved = sum(
        1
        for item in results
        if item.get("expected_document_best_rank") is not None
        and item["expected_document_best_rank"] <= 10
        and item["answerability"] != "CORPUS_GAP"
    )
    failed = sum(
        1
        for item in results
        if item["answerability"] != "CORPUS_GAP" and not item["passed_at_10"]
    )
    return {
        "corpus_gaps": corpus_gaps,
        "expected_document_retrieved_top10": expected_document_retrieved,
        "expected_chunk_retrieved_top10": expected_chunk_retrieved,
        "failed": failed,
    }


def _summarize_root_causes(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {label: 0 for label in [
        "CORPUS_GAP",
        "EXPECTED_DOCUMENT_WRONG_CHUNK",
        "RETRIEVAL_COMPETITION",
        "QUERY_MISMATCH",
        "EMBEDDING_REPRESENTATION",
        "RETRIEVAL_CONFIGURATION",
        "EVALUATION_MAPPING",
        "UNRESOLVED",
    ]}
    for item in results:
        label = item.get("root_cause")
        if label and label in summary:
            summary[label] += 1
    return summary


def format_phase15_text(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    passed = report["passed"]
    summary = report["summary"]
    root_causes = report["root_cause_summary"]
    lines = [
        "PHASE15_KB_V2_RETRIEVAL",
        "",
        f"Questions: {report['total_questions']}",
        "",
        f"Recall@1: {metrics['recall_at_1']}",
        f"Recall@3: {metrics['recall_at_3']}",
        f"Recall@5: {metrics['recall_at_5']}",
        f"Recall@10: {metrics['recall_at_10']}",
        f"MRR: {metrics['mrr']}",
        "",
        f"Passed@1: {passed['passed_at_1']}",
        f"Passed@3: {passed['passed_at_3']}",
        f"Passed@5: {passed['passed_at_5']}",
        f"Passed@10: {passed['passed_at_10']}",
        "",
        f"Corpus gaps: {summary['corpus_gaps']}",
        f"Expected document retrieved: {summary['expected_document_retrieved_top10']}",
        f"Expected chunk retrieved: {summary['expected_chunk_retrieved_top10']}",
        f"Failed: {summary['failed']}",
        "",
        "Root-cause summary:",
        f"CORPUS_GAP: {root_causes['CORPUS_GAP']}",
        f"EXPECTED_DOCUMENT_WRONG_CHUNK: {root_causes['EXPECTED_DOCUMENT_WRONG_CHUNK']}",
        f"RETRIEVAL_COMPETITION: {root_causes['RETRIEVAL_COMPETITION']}",
        f"QUERY_MISMATCH: {root_causes['QUERY_MISMATCH']}",
        f"EMBEDDING_REPRESENTATION: {root_causes['EMBEDDING_REPRESENTATION']}",
        f"RETRIEVAL_CONFIGURATION: {root_causes['RETRIEVAL_CONFIGURATION']}",
        f"EVALUATION_MAPPING: {root_causes['EVALUATION_MAPPING']}",
        f"UNRESOLVED: {root_causes['UNRESOLVED']}",
        "",
        f"Vector table: {report['vector_table']}",
        f"Embedding model: {report['embedding_model']}",
        f"Revision: {report['embedding_model_revision']}",
    ]
    lines.extend(["", "Per-query results:", ""])
    for query in report["queries"]:
        lines.extend(
            [
                f"{query['id']}: {query['question']}",
                f"  answerability={query['answerability']} root_cause={query.get('root_cause')}",
                f"  expected_chunk_rank={query.get('expected_chunk_rank')} "
                f"expected_document_best_rank={query.get('expected_document_best_rank')}",
                f"  recall@10={query['recall_at_10']} mrr={query['mrr']}",
                f"  mapping: {query.get('mapping_notes', '')}",
                "",
            ]
        )
    return "\n".join(lines)
