"""Ranking extraction helpers for retrieval diagnostics."""

from __future__ import annotations

from typing import Any

from app.kb.chunking.models import ProductionChunkRecord


def best_rank_for_ids(ranked_chunk_ids: list[str], target_ids: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in target_ids:
            return index
    return None


def best_rank_for_documents(ranked_hits: list[dict[str, Any]], document_ids: set[str]) -> tuple[int | None, float | None]:
    best_rank: int | None = None
    best_similarity: float | None = None
    for hit in ranked_hits:
        if hit.get("document_id") not in document_ids:
            continue
        rank = hit["rank"]
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_similarity = hit.get("similarity")
    return best_rank, best_similarity


def expected_chunk_rank_details(
    ranked_hits: list[dict[str, Any]],
    expected_chunk_ids: list[str],
) -> list[dict[str, Any]]:
    rank_by_id = {hit["chunk_id"]: hit for hit in ranked_hits}
    details: list[dict[str, Any]] = []
    for chunk_id in expected_chunk_ids:
        hit = rank_by_id.get(chunk_id)
        details.append(
            {
                "chunk_id": chunk_id,
                "rank": hit["rank"] if hit else None,
                "similarity": hit.get("similarity") if hit else None,
                "in_top_10": bool(hit and hit["rank"] <= 10),
                "in_top_20": bool(hit and hit["rank"] <= 20),
                "in_top_50": bool(hit and hit["rank"] <= 50),
                "in_top_100": bool(hit and hit["rank"] <= 100),
            }
        )
    return details


def summarize_expected_chunk_analysis(
    ranked_hits: list[dict[str, Any]],
    expected_chunk_ids: list[str],
) -> dict[str, Any]:
    details = expected_chunk_rank_details(ranked_hits, expected_chunk_ids)
    ranked = [item for item in details if item["rank"] is not None]
    best = min(ranked, key=lambda item: item["rank"]) if ranked else None
    return {
        "expected_chunk_ranks": details,
        "best_expected_chunk_rank": best["rank"] if best else None,
        "best_expected_chunk_similarity": best["similarity"] if best else None,
        "expected_chunks_in_top_10": any(item["in_top_10"] for item in details),
        "expected_chunks_in_top_20": any(item["in_top_20"] for item in details),
        "expected_chunks_in_top_50": any(item["in_top_50"] for item in details),
        "expected_chunks_in_top_100": any(item["in_top_100"] for item in details),
    }


def summarize_expected_document_analysis(
    ranked_hits: list[dict[str, Any]],
    expected_document_ids: list[str],
) -> dict[str, Any]:
    doc_ids = set(expected_document_ids)
    best_rank, best_similarity = best_rank_for_documents(ranked_hits, doc_ids)
    return {
        "expected_document_best_rank": best_rank,
        "expected_document_best_similarity": best_similarity,
        "expected_document_in_top_10": bool(best_rank is not None and best_rank <= 10),
        "expected_document_in_top_20": bool(best_rank is not None and best_rank <= 20),
        "expected_document_in_top_50": bool(best_rank is not None and best_rank <= 50),
        "expected_document_in_top_100": bool(best_rank is not None and best_rank <= 100),
    }


def format_vector_hit(
    *,
    rank: int,
    item: dict[str, Any],
    chunk: ProductionChunkRecord | None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": item["chunk_id"],
        "document_id": item.get("document_id") or (chunk.document_id if chunk else None),
        "title": item.get("title") or (chunk.title if chunk else ""),
        "canonical_url": item.get("canonical_url") or (chunk.canonical_url if chunk else ""),
        "document_type": item.get("document_type") or (chunk.document_type.value if chunk else ""),
        "source_type": item.get("source_type") or (chunk.source_type.value if chunk else ""),
        "section_path": item.get("section_path") or (chunk.section_path if chunk else []),
        "similarity": item.get("similarity"),
        "token_count": chunk.token_count if chunk else item.get("token_count", 0),
        "text": item.get("content") or (chunk.content if chunk else ""),
    }


def top_competing_chunks(
    ranked_hits: list[dict[str, Any]],
    *,
    expected_chunk_ids: set[str],
    expected_document_ids: set[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    competing: list[dict[str, Any]] = []
    for hit in ranked_hits:
        if hit["chunk_id"] in expected_chunk_ids:
            continue
        if hit.get("document_id") in expected_document_ids:
            continue
        competing.append(
            {
                "rank": hit["rank"],
                "chunk_id": hit["chunk_id"],
                "document_id": hit.get("document_id"),
                "similarity": hit.get("similarity"),
                "title": hit.get("title"),
                "canonical_url": hit.get("canonical_url"),
            }
        )
        if len(competing) >= limit:
            break
    return competing
