"""Match expected knowledge against ChatTrace retrieval rows. Does not retrieve again."""

from __future__ import annotations

from typing import Any

from app.evaluation.catalog import CatalogItem
from app.kb.evaluation.metrics import recall_at_k, reciprocal_rank


def candidate_ids(rows: list[dict[str, Any]] | None) -> list[str]:
    return [str(item.get("chunk_id")) for item in rows or [] if item.get("chunk_id")]


def retrieval_ids(trace: dict[str, Any]) -> list[str]:
    retrieval = trace.get("retrieval") or {}
    return candidate_ids(retrieval.get("candidates"))


def rerank_ids(trace: dict[str, Any]) -> list[str]:
    rerank = trace.get("reranking") or {}
    return candidate_ids(rerank.get("candidates"))


def final_ids(trace: dict[str, Any]) -> list[str]:
    context = trace.get("final_context") or {}
    return candidate_ids(context.get("chunks"))


def haystack(row: dict[str, Any]) -> str:
    section = row.get("section_path") or row.get("section") or ""
    if isinstance(section, list):
        section = " ".join(str(part) for part in section)
    parts = [
        str(row.get("chunk_id") or ""),
        str(row.get("title") or ""),
        str(section),
        str(row.get("text_preview") or row.get("text") or ""),
        str(row.get("knowledge_key") or ""),
    ]
    return " ".join(parts).lower()


def row_matches_item(row: dict[str, Any], item: CatalogItem) -> bool:
    if str(row.get("knowledge_key") or "") == item.knowledge_key:
        return True
    if item.knowledge_key and item.knowledge_key in str(row.get("chunk_id") or ""):
        return True
    text = haystack(row)
    anchor = item.anchor
    for exclude in anchor.exclude_patterns:
        if exclude.lower() in text:
            return False
    section_ok = True
    if anchor.section_path_contains:
        section_ok = any(part.lower() in text for part in anchor.section_path_contains)
    pattern_ok = True
    if anchor.content_patterns:
        pattern_ok = any(part.lower() in text for part in anchor.content_patterns)
    if anchor.section_path_contains or anchor.content_patterns:
        return section_ok and (pattern_ok if anchor.content_patterns else True)
    return item.knowledge_key.lower() in text


def matching_ranks(rows: list[dict[str, Any]], items: list[CatalogItem]) -> list[int]:
    ranks: list[int] = []
    for index, row in enumerate(rows or [], start=1):
        if any(row_matches_item(row, item) for item in items):
            ranks.append(index)
    return ranks


def expected_items_for_keys(keys: list[str], catalog: dict[str, CatalogItem]) -> list[CatalogItem]:
    return [catalog[key] for key in keys if key in catalog]


def evidence_in_rows(rows: list[dict[str, Any]], items: list[CatalogItem], *, k: int | None = None) -> bool:
    subset = list(rows or [])[:k] if k is not None else list(rows or [])
    if not items:
        return False
    return any(any(row_matches_item(row, item) for item in items) for row in subset)


def retrieval_metrics_from_ids(expected_ids: list[str], retrieved_ids: list[str]) -> dict[str, float]:
    return {
        "recall_at_1": recall_at_k(expected_ids, retrieved_ids, 1),
        "recall_at_3": recall_at_k(expected_ids, retrieved_ids, 3),
        "recall_at_5": recall_at_k(expected_ids, retrieved_ids, 5),
        "recall_at_10": recall_at_k(expected_ids, retrieved_ids, 10),
        "mrr": reciprocal_rank(expected_ids, retrieved_ids),
    }


def synthetic_expected_ids(ranks: list[int], retrieved_ids: list[str]) -> list[str]:
    """Map pattern matches onto IDs so existing recall@k / MRR helpers can be reused."""
    if not ranks:
        return ["expected-missing"]
    return [retrieved_ids[rank - 1] for rank in ranks if 0 < rank <= len(retrieved_ids)]
