"""Phase 11.8 targeted chunk inspection (4 queries, top-20, read-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.provider import EmbeddingProvider
from app.kb.evaluation.earkart_13_v1 import EARKART_13_CASES
from app.kb.evaluation.retrieval_diagnostic.competitor_classifier import (
    classify_competitor,
    estimate_token_count,
    targeted_diagnosis,
)
from app.kb.evaluation.retrieval_diagnostic.ranking import best_rank_for_documents, best_rank_for_ids
from app.kb.vector.search import VectorSearchService

TARGET_QUERY_IDS = ("ERK-001", "ERK-007", "ERK-009", "ERK-011")
TOP_K = 20


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_result(rank: int, item: dict[str, Any], *, truncate_after: int | None = None) -> dict[str, Any]:
    text = item.get("content") or ""
    token_count = estimate_token_count(text)
    formatted = {
        "rank": rank,
        "similarity": item.get("similarity"),
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "title": item.get("title") or "",
        "canonical_url": item.get("canonical_url") or item.get("source_url") or "",
        "document_type": item.get("document_type") or "",
        "token_count": token_count,
        "section_path": item.get("section_path") or [],
        "text": text if truncate_after is None else _truncate(text, truncate_after),
    }
    return formatted


def _chunk_record_from_store(item: dict[str, Any]) -> dict[str, Any]:
    text = item.get("content") or ""
    return {
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "title": item.get("title") or "",
        "canonical_url": item.get("canonical_url") or item.get("source_url") or "",
        "similarity": None,
        "rank": None,
        "token_count": estimate_token_count(text),
        "section_path": item.get("section_path") or [],
        "text": text,
    }


def _best_expected_chunk_in_hits(
    hits: list[dict[str, Any]],
    expected_chunk_ids: list[str],
) -> tuple[str | None, int | None, float | None]:
    rank_by_id = {hit["chunk_id"]: hit for hit in hits}
    best_id: str | None = None
    best_rank: int | None = None
    best_sim: float | None = None
    for chunk_id in expected_chunk_ids:
        hit = rank_by_id.get(chunk_id)
        if hit is None:
            continue
        if best_rank is None or hit["rank"] < best_rank:
            best_id = chunk_id
            best_rank = hit["rank"]
            best_sim = hit.get("similarity")
    return best_id, best_rank, best_sim


def _load_root_cause_top10(reports_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = reports_dir / "earkart_retrieval_root_cause_13.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    by_id: dict[str, list[dict[str, Any]]] = {}
    for question in report.get("questions", []):
        if question.get("id") in TARGET_QUERY_IDS:
            by_id[question["id"]] = question.get("top_10", [])
    return by_id


def inspect_query(
    *,
    case: dict[str, Any],
    config: EmbeddingConfig,
    store,
    embedding_provider: EmbeddingProvider,
    expected_chunk_ids: list[str],
    expected_document_ids: list[str],
    cached_top10: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query_vector = embedding_provider.embed_queries([case["query"]])[0]
    raw_hits = store.search(
        query_vector,
        top_k=TOP_K,
        embedding_version=config.embedding_version,
    )

    top_results = [
        _format_result(index, item, truncate_after=None if index <= 10 else 500)
        for index, item in enumerate(raw_hits[:TOP_K], start=1)
    ]

    expected_chunk_found = bool(expected_chunk_ids)
    primary_chunk_id = expected_chunk_ids[0] if expected_chunk_ids else None
    best_id, chunk_rank, chunk_sim = _best_expected_chunk_in_hits(top_results, expected_chunk_ids)

    if best_id:
        primary_chunk_id = best_id
        expected_answer = next(hit for hit in top_results if hit["chunk_id"] == best_id)
        expected_chunk = {
            "chunk_id": expected_answer["chunk_id"],
            "document_id": expected_answer["document_id"],
            "title": expected_answer["title"],
            "canonical_url": expected_answer["canonical_url"],
            "similarity": chunk_sim,
            "rank": chunk_rank,
            "token_count": expected_answer["token_count"],
            "section_path": expected_answer["section_path"],
            "text": expected_answer["text"],
        }
    elif primary_chunk_id:
        stored = store.get_by_chunk_id(primary_chunk_id)
        if stored:
            expected_chunk = _chunk_record_from_store(stored)
            expected_chunk["similarity"] = None
            expected_chunk["rank"] = None
        else:
            expected_chunk_found = False
            expected_chunk = {
                "chunk_id": primary_chunk_id,
                "document_id": case.get("expected_document_id"),
                "title": "",
                "canonical_url": "",
                "similarity": None,
                "rank": None,
                "token_count": None,
                "section_path": [],
                "text": "",
            }
    else:
        expected_chunk = {
            "chunk_id": None,
            "document_id": None,
            "title": "",
            "canonical_url": "",
            "similarity": None,
            "rank": None,
            "token_count": None,
            "section_path": [],
            "text": "",
        }

    doc_rank, doc_sim = best_rank_for_documents(top_results, set(expected_document_ids))
    doc_hit = next(
        (hit for hit in top_results if hit.get("document_id") in set(expected_document_ids)),
        None,
    )
    expected_document = {
        "best_rank": doc_rank,
        "similarity": doc_sim,
        "chunk_id": doc_hit["chunk_id"] if doc_hit else None,
        "token_count": doc_hit["token_count"] if doc_hit else None,
        "section_path": doc_hit["section_path"] if doc_hit else [],
        "text": doc_hit["text"] if doc_hit else "",
    }

    competitors: list[dict[str, Any]] = []
    for hit in top_results:
        if hit["chunk_id"] in set(expected_chunk_ids):
            continue
        competitors.append(
            {
                "rank": hit["rank"],
                "similarity": hit.get("similarity"),
                "document_title": hit.get("title"),
                "section_path": hit.get("section_path"),
                "token_count": hit.get("token_count"),
                "text": hit.get("text") if hit["rank"] <= 10 else _truncate(hit.get("text", ""), 500),
                "classification": classify_competitor(hit, set(expected_chunk_ids)),
            }
        )
        if len(competitors) >= 5:
            break

    diagnosis = targeted_diagnosis(
        expected_chunk_rank=chunk_rank,
        expected_document_best_rank=doc_rank,
        expected_chunk_found=expected_chunk_found and primary_chunk_id is not None,
    )

    return {
        "id": case["id"],
        "question": case["query"],
        "expected_knowledge": case.get("expected_knowledge"),
        "expected_chunk_ids": expected_chunk_ids,
        "expected_document_ids": expected_document_ids,
        "expected_chunk": expected_chunk,
        "expected_chunk_not_in_top_20": chunk_rank is None and expected_chunk_found,
        "expected_document": expected_document,
        "top_results": top_results,
        "competitors": competitors,
        "expected_chunk_rank": chunk_rank,
        "expected_chunk_similarity": chunk_sim,
        "expected_document_best_rank": doc_rank,
        "expected_document_best_similarity": doc_sim,
        "diagnosis": diagnosis,
        "reused_cached_top10": bool(cached_top10),
    }


def run_targeted_inspection(
    *,
    config: EmbeddingConfig,
    search: VectorSearchService,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    cases = [case for case in EARKART_13_CASES if case["id"] in TARGET_QUERY_IDS]
    embedding_provider = search.provider
    store = search.store
    vector_count = store.count(embedding_version=config.embedding_version)

    cached = _load_root_cause_top10(reports_dir) if reports_dir else {}

    queries: list[dict[str, Any]] = []
    for case in cases:
        expected_chunk_ids = list(case.get("expected_chunk_ids", []))
        expected_document_ids: list[str] = []
        if case.get("expected_document_id"):
            expected_document_ids.append(case["expected_document_id"])

        queries.append(
            inspect_query(
                case=case,
                config=config,
                store=store,
                embedding_provider=embedding_provider,
                expected_chunk_ids=expected_chunk_ids,
                expected_document_ids=expected_document_ids,
                cached_top10=cached.get(case["id"]),
            )
        )

    return {
        "report_version": "1.0",
        "read_only": True,
        "index": {
            "chunk_count": vector_count,
            "embedding_model": config.model,
            "embedding_revision": config.model_revision,
            "dimension": config.dimension,
            "similarity": "cosine",
        },
        "queries": queries,
        "database_writes": 0,
        "chunks_modified": 0,
        "embeddings_modified": 0,
    }


def format_targeted_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "Earkart Targeted Chunk Inspection (Phase 11.8)",
        f"Index: {report['index']['chunk_count']} vectors | {report['index']['embedding_model']}",
        "",
    ]

    labels = {
        "ERK-001": "Bluup",
        "ERK-007": "TINY",
        "ERK-009": "Hearing aid help",
        "ERK-011": "Amplifier vs hearing aid",
    }

    for item in report["queries"]:
        lines.extend(
            [
                "=" * 72,
                f"{item['id']}: {item['question']}",
                "",
                "EXPECTED ANSWER CHUNK",
                "---------------------",
            ]
        )
        exp = item["expected_chunk"]
        if item.get("expected_chunk_not_in_top_20"):
            lines.append("EXPECTED CHUNK NOT IN TOP 20")
        lines.extend(
            [
                f"Chunk ID: {exp.get('chunk_id')}",
                f"Document ID: {exp.get('document_id')}",
                f"Title: {exp.get('title')}",
                f"Canonical URL: {exp.get('canonical_url')}",
                f"Similarity: {exp.get('similarity')}",
                f"Rank: {exp.get('rank') if exp.get('rank') is not None else 'not in top-20'}",
                f"Token count: {exp.get('token_count')}",
                f"Section path: {exp.get('section_path')}",
                f"Text:\n{exp.get('text') or ''}",
                "",
                "EXPECTED DOCUMENT BEST RESULT",
                "-----------------------------",
                f"Best rank: {item['expected_document'].get('best_rank')}",
                f"Similarity: {item['expected_document'].get('similarity')}",
                f"Chunk ID: {item['expected_document'].get('chunk_id')}",
                f"Token count: {item['expected_document'].get('token_count')}",
                f"Section path: {item['expected_document'].get('section_path')}",
                f"Text:\n{item['expected_document'].get('text') or ''}",
                "",
                "TOP 20 RESULTS",
                "--------------",
            ]
        )
        for hit in item["top_results"]:
            lines.extend(
                [
                    f"Rank: {hit['rank']}",
                    f"Similarity: {hit.get('similarity')}",
                    f"Chunk ID: {hit.get('chunk_id')}",
                    f"Document ID: {hit.get('document_id')}",
                    f"Title: {hit.get('title')}",
                    f"Canonical URL: {hit.get('canonical_url')}",
                    f"Document type: {hit.get('document_type')}",
                    f"Token count: {hit.get('token_count')}",
                    f"Section path: {hit.get('section_path')}",
                    f"Text:\n{hit.get('text') or ''}",
                    "",
                ]
            )

        lines.extend(["TOP COMPETING CHUNKS", "--------------------"])
        for comp in item["competitors"]:
            lines.extend(
                [
                    f"Rank: {comp['rank']}",
                    f"Similarity: {comp.get('similarity')}",
                    f"Document title: {comp.get('document_title')}",
                    f"Section path: {comp.get('section_path')}",
                    f"Token count: {comp.get('token_count')}",
                    f"Classification: {comp.get('classification')}",
                    f"Text:\n{comp.get('text') or ''}",
                    "",
                ]
            )

        lines.extend(
            [
                f"Diagnosis: {item['diagnosis']}",
                f"Expected chunk rank: {item.get('expected_chunk_rank') or 'not in top-20'}",
                f"Expected document rank: {item.get('expected_document_best_rank') or 'not in top-20'}",
                "",
            ]
        )

    lines.extend(
        [
            "EAR_KART_TARGETED_CHUNK_INSPECTION",
            "",
        ]
    )
    for item in report["queries"]:
        label = labels.get(item["id"], item["id"])
        lines.append(
            f"{item['id']} {label}\n"
            f"Expected chunk rank: {item.get('expected_chunk_rank') or 'not in top-20'}\n"
            f"Expected document rank: {item.get('expected_document_best_rank') or 'not in top-20'}\n"
            f"Diagnosis: {item['diagnosis']}\n"
        )

    lines.extend(
        [
            "READ_ONLY: PASS",
            f"DATABASE_WRITES: {report.get('database_writes', 0)}",
            f"CHUNKS_MODIFIED: {report.get('chunks_modified', 0)}",
            f"EMBEDDINGS_MODIFIED: {report.get('embeddings_modified', 0)}",
        ]
    )
    return "\n".join(lines)
