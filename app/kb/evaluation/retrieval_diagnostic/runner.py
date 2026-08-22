"""Orchestrator for Earkart retrieval root-cause diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.loader import ChunkLoader
from app.kb.embedding.provider import EmbeddingProvider
from app.kb.evaluation.earkart_13_v1 import EARKART_13_CASES, EVALUATION_DATASET_VERSION
from app.kb.evaluation.retrieval_diagnostic.bm25 import Bm25Index
from app.kb.evaluation.retrieval_diagnostic.corpus_check import ROOT_CAUSE_LABELS, resolve_expected_status
from app.kb.evaluation.retrieval_diagnostic.ranking import (
    best_rank_for_documents,
    best_rank_for_ids,
    format_vector_hit,
    summarize_expected_chunk_analysis,
    summarize_expected_document_analysis,
)
from app.kb.evaluation.retrieval_diagnostic.root_cause import classify_root_cause
from app.kb.evaluation.retrieval_diagnostic.whole_document import WholeDocumentIndex, build_document_texts
from app.kb.vector.search import VectorSearchService
from app.kb.vector.store import VectorStore


def run_root_cause_diagnostic(
    *,
    config: EmbeddingConfig,
    chunks: list[ProductionChunkRecord],
    search: VectorSearchService,
    provider: EmbeddingProvider | None = None,
    top_k: int = 100,
    progress: bool = False,
) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    known_chunk_ids = set(chunk_by_id)
    embedding_provider = provider or search.provider
    bm25 = Bm25Index(chunks)
    document_texts = build_document_texts(chunks)
    whole_document_index = WholeDocumentIndex(provider=embedding_provider, document_texts=document_texts)

    store = search.store
    vector_count = store.count(embedding_version=config.embedding_version)

    questions: list[dict[str, Any]] = []
    total = len(EARKART_13_CASES)
    for index, case in enumerate(EARKART_13_CASES, start=1):
        if progress:
            print(f"[{index}/{total}] {case['id']}: vector top-{top_k} + BM25 + whole-doc...", flush=True)
        expected = resolve_expected_status(
            case=case,
            chunks=chunks,
            chunk_by_id=chunk_by_id,
            known_chunk_ids=known_chunk_ids,
        )
        query_vectors = embedding_provider.embed_queries([case["query"]])
        query_vector = query_vectors[0]
        raw_hits = store.search(
            query_vector,
            top_k=top_k,
            embedding_version=config.embedding_version,
        )
        vector_hits = [
            format_vector_hit(rank=index, item=item, chunk=chunk_by_id.get(item["chunk_id"]))
            for index, item in enumerate(raw_hits[:top_k], start=1)
        ]

        expected_chunk_ids = [
            chunk_id for chunk_id in expected["expected_chunk_ids"] if chunk_id in known_chunk_ids
        ]
        chunk_analysis = summarize_expected_chunk_analysis(vector_hits, expected_chunk_ids)
        document_analysis = summarize_expected_document_analysis(
            vector_hits,
            expected["expected_document_ids"],
        )

        keyword_hits = bm25.search(case["query"], top_k=top_k)
        keyword_chunk_rank = best_rank_for_ids(
            [item["chunk_id"] for item in keyword_hits],
            set(expected_chunk_ids),
        )
        keyword_doc_rank, _keyword_doc_sim = best_rank_for_documents(
            [
                {
                    "rank": item["rank"],
                    "document_id": item["document_id"],
                    "similarity": item.get("score"),
                }
                for item in keyword_hits
            ],
            set(expected["expected_document_ids"]),
        )
        keyword_analysis = {
            "method": "bm25",
            "best_rank": keyword_hits[0]["rank"] if keyword_hits else None,
            "expected_chunk_best_rank": keyword_chunk_rank,
            "expected_document_best_rank": keyword_doc_rank,
            "top_10": keyword_hits[:10],
            "top_100": keyword_hits[:top_k],
        }

        if progress and expected["expected_document_ids"]:
            print(
                f"    whole-doc: embedding {len(expected['expected_document_ids'])} expected document(s)...",
                flush=True,
            )
        whole_document_analysis = whole_document_index.analyze(
            query_vector,
            expected["expected_document_ids"],
            vector_hits=vector_hits,
        )
        whole_document_analysis.pop("document_similarities", None)
        whole_document_analysis.pop("rank_comparison_pool_size", None)

        diagnosis = classify_root_cause(
            case=case,
            expected=expected,
            chunk_analysis=chunk_analysis,
            document_analysis=document_analysis,
            keyword_analysis=keyword_analysis,
            whole_document_analysis=whole_document_analysis,
            vector_hits=vector_hits,
            retrieval_config_issues=_retrieval_config_issues(vector_count, len(chunks)),
        )

        questions.append(
            {
                "id": case["id"],
                "question": case["query"],
                "category": case.get("category"),
                "expected": expected,
                "vector_search": {
                    "query_embedding_model": config.model,
                    "top_100": vector_hits,
                },
                "expected_chunk_analysis": chunk_analysis,
                "expected_document_analysis": document_analysis,
                "keyword_analysis": keyword_analysis,
                "whole_document_analysis": whole_document_analysis,
                "diagnosis": diagnosis,
            }
        )

    summary = _build_summary(questions)
    final_verdict = _final_verdict(questions, summary)
    return {
        "report_version": "1.0",
        "report_type": "earkart_retrieval_root_cause_13",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "dataset": EVALUATION_DATASET_VERSION,
            "question_count": len(EARKART_13_CASES),
            "index": {
                "chunk_count": vector_count,
                "embedding_model": config.model,
                "embedding_revision": config.model_revision,
                "embedding_dimension": config.dimension,
                "vector_store": "pgvector",
                "similarity_metric": "cosine",
            },
            "retrieval": {"top_k": top_k},
        },
        "summary": summary,
        "questions": questions,
        "final_verdict": final_verdict,
    }


def _retrieval_config_issues(vector_count: int, manifest_count: int) -> list[str]:
    issues: list[str] = []
    if vector_count != manifest_count:
        issues.append(
            f"Production vector count ({vector_count}) differs from frozen manifest count ({manifest_count})."
        )
    return issues


def _build_summary(questions: list[dict[str, Any]]) -> dict[str, Any]:
    with_expected = sum(1 for item in questions if item["expected"]["expected_chunk_ids"])
    without_expected = len(questions) - with_expected
    root_cause_counts = {label: 0 for label in ROOT_CAUSE_LABELS}

    def count_flag(key: str) -> int:
        return sum(1 for item in questions if item["expected_chunk_analysis"].get(key))

    def count_doc_flag(key: str) -> int:
        return sum(1 for item in questions if item["expected_document_analysis"].get(key))

    for item in questions:
        primary = item["diagnosis"]["primary_root_cause"]
        root_cause_counts[primary] = root_cause_counts.get(primary, 0) + 1

    return {
        "questions_with_expected_chunks": with_expected,
        "questions_without_expected_chunks": without_expected,
        "expected_chunks_found_in_top_10": count_flag("expected_chunks_in_top_10"),
        "expected_chunks_found_in_top_20": count_flag("expected_chunks_in_top_20"),
        "expected_chunks_found_in_top_50": count_flag("expected_chunks_in_top_50"),
        "expected_chunks_found_in_top_100": count_flag("expected_chunks_in_top_100"),
        "expected_documents_found_in_top_10": count_doc_flag("expected_document_in_top_10"),
        "expected_documents_found_in_top_20": count_doc_flag("expected_document_in_top_20"),
        "expected_documents_found_in_top_50": count_doc_flag("expected_document_in_top_50"),
        "expected_documents_found_in_top_100": count_doc_flag("expected_document_in_top_100"),
        "root_cause_counts": root_cause_counts,
    }


def _final_verdict(questions: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    failed = [
        item
        for item in questions
        if not item["expected_chunk_analysis"].get("expected_chunks_in_top_10")
    ]
    if not failed:
        return "ROOT_CAUSE_IDENTIFIED"
    unresolved_failed = sum(
        1 for item in failed if item["diagnosis"]["primary_root_cause"] == "UNRESOLVED"
    )
    if unresolved_failed >= max(2, len(failed) // 2):
        return "NEEDS_MORE_EVIDENCE"
    return "ROOT_CAUSE_IDENTIFIED"


