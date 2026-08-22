"""Retrieval-only evaluation for the 13-question Earkart benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from app.kb.evaluation.earkart_13_v1 import EARKART_13_CASES, EVALUATION_DATASET_VERSION
from app.kb.vector.search import VectorSearchService
from app.kb.vector.store import VectorStore


def _expected_rank(expected_ids: list[str], retrieved_ids: list[str]) -> int | None:
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_ids:
            return index
    return None


def _categorize_failure(
    *,
    case: dict[str, Any],
    retrieved: list[dict[str, Any]],
    expected_ids: list[str],
    known_chunk_ids: set[str],
) -> str:
    if case.get("failure_category_default"):
        return case["failure_category_default"]
    if not expected_ids:
        return "EVALUATION_MAPPING"
    missing = [chunk_id for chunk_id in expected_ids if chunk_id not in known_chunk_ids]
    if missing:
        return "EVALUATION_MAPPING"
    rank = _expected_rank(expected_ids, [item["chunk_id"] for item in retrieved])
    if rank is None:
        top_urls = " ".join(item.get("source_url", "") for item in retrieved[:3]).lower()
        if any(token in top_urls for token in ("investor", "prospectus", "transcript", "press-release")):
            if case.get("category") in {"product", "product_pricing", "product_education", "support"}:
                return "RETRIEVAL"
        return "EMBEDDING"
    if rank <= 3:
        return "QUERY"
    return "RETRIEVAL"


def _retrieval_verdict(metrics: dict[str, Any], results: list[dict[str, Any]]) -> str:
    if metrics["recall_at_10"] >= 0.7 and metrics["mrr"] >= 0.5:
        return "PASS"
    source_data_failures = sum(1 for item in results if item.get("failure_category") == "SOURCE_DATA")
    if metrics["recall_at_10"] < 0.35 and source_data_failures <= 1:
        return "FAIL"
    return "NEEDS_INVESTIGATION"


def run_evaluation(*, top_k: int = 10) -> dict[str, Any]:
    config = EmbeddingConfig.from_settings()
    loader = ChunkLoader(settings.chunks_dir, config.embedding_input_manifest)
    chunks = loader.load_all().chunks
    known_chunk_ids = {chunk.chunk_id for chunk in chunks}
    chunk_meta = {chunk.chunk_id: chunk for chunk in chunks}

    store = VectorStore(config.database_url, config.vector_table, config.dimension)
    vector_count = store.count(embedding_version=config.embedding_version)
    search = VectorSearchService(config=config, store=store)

    per_query: list[dict[str, Any]] = []
    for case in EARKART_13_CASES:
        expected_ids = [
            chunk_id for chunk_id in case["expected_chunk_ids"] if chunk_id in known_chunk_ids
        ]
        require_all = False
        results = search.search(case["query"], top_k=top_k)
        retrieved_ids = [item["chunk_id"] for item in results]

        formatted_results: list[dict[str, Any]] = []
        for index, item in enumerate(results[:top_k], start=1):
            chunk = chunk_meta.get(item["chunk_id"])
            formatted_results.append(
                {
                    "rank": index,
                    "similarity": item.get("similarity"),
                    "distance": item.get("distance"),
                    "chunk_id": item["chunk_id"],
                    "document_id": item.get("document_id"),
                    "title": item.get("title") or (chunk.title if chunk else None),
                    "source_url": item.get("source_url") or (chunk.source_url if chunk else None),
                    "section_path": item.get("section_path") or (chunk.section_path if chunk else []),
                    "content": item.get("content") or (chunk.content if chunk else ""),
                }
            )

        expected_rank = _expected_rank(expected_ids, retrieved_ids)
        passed_at = {
            f"passed_at_{k}": recall_at_k(expected_ids, retrieved_ids, k, require_all=require_all) == 1.0
            for k in (1, 3, 5, 10)
        }
        passed = passed_at[f"passed_at_{top_k}"]

        failure_category = None
        if not passed:
            failure_category = _categorize_failure(
                case=case,
                retrieved=formatted_results,
                expected_ids=expected_ids,
                known_chunk_ids=known_chunk_ids,
            )

        expected_docs = []
        for chunk_id in expected_ids:
            chunk = chunk_meta[chunk_id]
            expected_docs.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source_url": chunk.source_url,
                    "section_path": chunk.section_path,
                }
            )

        per_query.append(
            {
                "id": case["id"],
                "query": case["query"],
                "category": case.get("category"),
                "query_note": case.get("query_note"),
                "expected_knowledge": case["expected_knowledge"],
                "expected_document_id": case.get("expected_document_id"),
                "expected_chunks": expected_docs,
                "expected_chunk_ids": expected_ids,
                "expected_rank": expected_rank,
                "retrieved": formatted_results,
                "retrieved_chunk_ids": retrieved_ids,
                "recall_at_1": recall_at_k(expected_ids, retrieved_ids, 1, require_all=require_all),
                "recall_at_3": recall_at_k(expected_ids, retrieved_ids, 3, require_all=require_all),
                "recall_at_5": recall_at_k(expected_ids, retrieved_ids, 5, require_all=require_all),
                "recall_at_10": recall_at_k(expected_ids, retrieved_ids, 10, require_all=require_all),
                "mrr": reciprocal_rank(expected_ids, retrieved_ids, require_all=require_all),
                **passed_at,
                "passed": passed,
                "failure_category": failure_category,
            }
        )

    metrics = aggregate_metrics(per_query)
    metrics["passed_at_1"] = sum(1 for item in per_query if item["passed_at_1"])
    metrics["passed_at_3"] = sum(1 for item in per_query if item["passed_at_3"])
    metrics["passed_at_5"] = sum(1 for item in per_query if item["passed_at_5"])
    metrics["passed_at_10"] = sum(1 for item in per_query if item["passed_at_10"])

    verdict = _retrieval_verdict(metrics, per_query)
    return {
        "evaluation_dataset_version": EVALUATION_DATASET_VERSION,
        "embedding_version": config.embedding_version,
        "embedding_model": config.model,
        "embedding_model_revision": config.model_revision,
        "embedding_dimension": config.dimension,
        "vector_table": config.vector_table,
        "vector_count": vector_count,
        "kb_dataset_version": config.kb_dataset_version,
        "metrics": metrics,
        "retrieval_quality_verdict": verdict,
        "total_evaluation_queries": metrics["total"],
        "passed_queries": metrics["passed"],
        "results": per_query,
        "passed": verdict == "PASS",
    }


def _truncate(text: str, limit: int = 400) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _format_text_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "Earkart 13 Retrieval Evaluation",
        f"Dataset: {report['evaluation_dataset_version']}",
        f"Embedding model: {report['embedding_model']}",
        f"Embedding revision: {report['embedding_model_revision']}",
        f"Embedding version: {report['embedding_version']}",
        f"Vector table: {report['vector_table']}",
        f"Vectors in index: {report['vector_count']}",
        "",
        f"Recall@1: {metrics['recall_at_1']}",
        f"Recall@3: {metrics['recall_at_3']}",
        f"Recall@5: {metrics['recall_at_5']}",
        f"Recall@10: {metrics['recall_at_10']}",
        f"MRR: {metrics['mrr']}",
        "",
        f"Passed @1: {metrics['passed_at_1']}/{metrics['total']}",
        f"Passed @3: {metrics['passed_at_3']}/{metrics['total']}",
        f"Passed @5: {metrics['passed_at_5']}/{metrics['total']}",
        f"Passed @10: {metrics['passed_at_10']}/{metrics['total']}",
        "",
        "Per-question results:",
    ]
    for item in report["results"]:
        lines.extend(
            [
                "",
                f"=== {item['id']}: {item['query']} ===",
                f"Expected knowledge: {item['expected_knowledge']}",
                f"Expected chunks: {[chunk['chunk_id'] for chunk in item['expected_chunks']]}",
                f"Expected rank: {item['expected_rank'] or 'not in top-10'}",
                f"Passed @10: {item['passed']}",
            ]
        )
        for hit in item["retrieved"]:
            lines.append(
                "  "
                + f"#{hit['rank']} sim={hit.get('similarity')} dist={hit.get('distance')} "
                + f"chunk={hit['chunk_id'][:16]}... doc={hit.get('document_id')} "
                + f"title={hit.get('title')!r} url={hit.get('source_url')}"
            )
            lines.append(f"    section_path={hit.get('section_path')}")
            lines.append(f"    text={_truncate(hit.get('content', ''))}")

    lines.extend(["", "Failed questions:"])
    for item in report["results"]:
        if item["passed"]:
            continue
        lines.extend(
            [
                f"- {item['id']}: {item['query']}",
                f"  expected knowledge: {item['expected_knowledge']}",
                f"  expected chunks: {[chunk['chunk_id'] for chunk in item['expected_chunks']]}",
                f"  expected rank: {item['expected_rank'] or 'not in top-10'}",
                f"  failure category: {item.get('failure_category')}",
            ]
        )
    lines.extend(
        [
            "",
            "EAR_KART_13_RETRIEVAL_EVALUATION",
            "",
            f"Recall@1: {metrics['recall_at_1']}",
            f"Recall@3: {metrics['recall_at_3']}",
            f"Recall@5: {metrics['recall_at_5']}",
            f"Recall@10: {metrics['recall_at_10']}",
            f"MRR: {metrics['mrr']}",
            f"Passed @1: {metrics['passed_at_1']}/{metrics['total']}",
            f"Passed @3: {metrics['passed_at_3']}/{metrics['total']}",
            f"Passed @5: {metrics['passed_at_5']}/{metrics['total']}",
            f"Passed @10: {metrics['passed_at_10']}/{metrics['total']}",
            "",
            f"RETRIEVAL_QUALITY: {report['retrieval_quality_verdict']}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on 13 Earkart benchmark questions.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = run_evaluation(top_k=args.top_k)
    json_path = reports_dir / "earkart_13_retrieval_evaluation.json"
    txt_path = reports_dir / "earkart_13_retrieval_evaluation.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(_format_text_report(report), encoding="utf-8")
    print(_format_text_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
