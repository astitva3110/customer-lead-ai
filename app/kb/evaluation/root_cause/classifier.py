"""Unified root-cause classification for retrieval evaluation."""

from __future__ import annotations

from typing import Any

ROOT_CAUSE_LABELS = [
    "CORPUS_GAP",
    "EXPECTED_CHUNK_RETRIEVED",
    "EXPECTED_DOCUMENT_WRONG_CHUNK",
    "CHUNK_GRANULARITY",
    "CHUNK_FRAGMENTATION",
    "RETRIEVAL_COMPETITION",
    "QUERY_MISMATCH",
    "EMBEDDING_REPRESENTATION",
    "RETRIEVAL_CONFIGURATION",
    "EVALUATION_MAPPING",
    "SUMMARY_MISSING",
    "UNRESOLVED",
]


def classify_root_cause(
    *,
    answerability: str,
    expected_chunk_ids: list[str],
    expected_chunk_rank: int | None,
    expected_document_best_rank: int | None,
    passed_at_10: bool,
    top_results: list[dict[str, Any]],
    knowledge_key: str | None = None,
    expected_chunk_count: int = 1,
) -> str | None:
    if passed_at_10:
        return None
    if answerability == "CORPUS_GAP":
        return "CORPUS_GAP"
    if not expected_chunk_ids:
        return "EVALUATION_MAPPING"

    if expected_chunk_rank is not None and expected_chunk_rank <= 10:
        return None

    if knowledge_key == "benefits:why_choose_summary" and expected_chunk_rank and expected_chunk_rank > 10:
        return "SUMMARY_MISSING"

    if expected_document_best_rank is None:
        return "CORPUS_GAP"

    top10 = top_results[:10]
    expected_docs = {hit.get("document_id") for hit in top_results if hit.get("chunk_id") in expected_chunk_ids}
    top10_has_expected_doc = any(hit.get("document_id") in expected_docs for hit in top10)
    top10_has_expected_chunk = any(hit.get("chunk_id") in expected_chunk_ids for hit in top10)

    if top10_has_expected_doc and not top10_has_expected_chunk:
        if expected_chunk_count > 3:
            return "CHUNK_FRAGMENTATION"
        top10_tokens = [
            hit.get("token_count") or 0
            for hit in top10
            if hit.get("document_id") in expected_docs and hit.get("chunk_id") not in expected_chunk_ids
        ]
        if top10_tokens and max(top10_tokens, default=0) < 40:
            return "CHUNK_GRANULARITY"
        return "EXPECTED_DOCUMENT_WRONG_CHUNK"

    if expected_document_best_rank <= 100:
        for hit in top10:
            if hit.get("chunk_id") in expected_chunk_ids:
                continue
            if hit.get("document_id") not in expected_docs:
                return "RETRIEVAL_COMPETITION"
        return "EMBEDDING_REPRESENTATION"

    return "UNRESOLVED"
