"""Score fused candidate lists against catalog anchors. Eval only."""

from __future__ import annotations

from typing import Any

from app.services.retrieval.models import RetrievalCandidate
from app.evaluation.catalog import CatalogItem
from app.evaluation.evidence import matching_ranks, row_matches_item
from app.evaluation.fusion import FusionConfig, StrategyName, fuse


def candidate_as_row(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "chunk_id": candidate.chunk_id,
        "title": candidate.document_title,
        "section_path": candidate.section_path,
        "text": candidate.text,
        "text_preview": (candidate.text or "")[:220],
        "vector_score": candidate.vector_score,
        "keyword_score": candidate.keyword_score,
    }


def _section_label(row: RetrievalCandidate) -> str:
    path = " / ".join(str(part) for part in row.section_path)
    return path or row.document_title


def catalog_rank(rows: list[RetrievalCandidate], item: CatalogItem) -> int | None:
    matches = matching_ranks([candidate_as_row(row) for row in rows], [item])
    return matches[0] if matches else None


def first_match_section(rows: list[RetrievalCandidate], item: CatalogItem) -> str | None:
    for row in rows:
        if matches_catalog(row, item):
            return _section_label(row)
    return None


def matches_catalog(candidate: RetrievalCandidate, item: CatalogItem) -> bool:
    return row_matches_item(candidate_as_row(candidate), item)


def list_metrics(rank: int | None, *, k: int = 10) -> dict[str, Any]:
    hit = rank is not None and rank <= k
    return {
        "rank": rank,
        "hit_at_10": hit,
        "mrr": 0.0 if rank is None else round(1.0 / rank, 4),
    }


def compare_lists(
    vector: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    item: CatalogItem,
    config: FusionConfig,
    strategies: list[StrategyName],
) -> dict[str, Any]:
    vector_rank = catalog_rank(vector, item)
    keyword_rank = catalog_rank(keyword, item)
    payload: dict[str, Any] = {
        "vector": {
            **list_metrics(vector_rank, k=config.top_k),
            "pool_size": len(vector),
            "pool_rank": vector_rank,
            "matched_section": first_match_section(vector, item),
            "top_sections": [_section_label(row) for row in vector[:10]],
        },
        "keyword": {
            **list_metrics(keyword_rank, k=config.top_k),
            "pool_size": len(keyword),
            "pool_rank": keyword_rank,
            "matched_section": first_match_section(keyword, item),
            "top_sections": [_section_label(row) for row in keyword[:10]],
        },
        "strategies": {},
    }
    for name in strategies:
        fused = fuse(name, vector, keyword, config)
        rank = catalog_rank(fused, item)
        payload["strategies"][name] = {
            **list_metrics(rank, k=config.top_k),
            "matched_section": first_match_section(fused, item),
            "top_sections": [_section_label(row) for row in fused[:5]],
        }
    in_vector = vector_rank is not None
    in_keyword = keyword_rank is not None
    current_hit = bool(payload["strategies"].get("current_merge", {}).get("hit_at_10"))
    if in_vector and not current_hit:
        locus = "merge_only"
    elif not in_vector and not in_keyword:
        locus = "absent_from_both_pools"
    elif not in_vector and in_keyword:
        locus = "keyword_only"
    elif in_vector:
        locus = "vector_has_it"
    else:
        locus = "unknown"
    payload["failure_locus"] = locus
    payload["in_vector_pool"] = in_vector
    payload["in_keyword_pool"] = in_keyword
    return payload
