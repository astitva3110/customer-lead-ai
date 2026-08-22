from __future__ import annotations

from typing import Any

from app.services.retrieval.models import RetrievalCandidate
from app.kb.evaluation.models import RankedHit


def _estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def metadata_from_store_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_number": item.get("page_number"),
        "document_title": item.get("title") or "",
        "section_path": list(item.get("section_path") or []),
        "content_type": item.get("split_method") or item.get("source_type") or "paragraph",
        "token_count": _estimate_token_count(item.get("content") or ""),
        "document_version": item.get("document_version"),
        "source_url": item.get("source_url") or "",
        "canonical_url": item.get("canonical_url") or "",
    }


def candidate_from_store_item(
    item: dict[str, Any],
    *,
    vector_score: float | None = None,
    keyword_score: float | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=item["chunk_id"],
        document_id=item.get("document_id") or "",
        text=item.get("content") or "",
        metadata=metadata_from_store_item(item),
        vector_score=vector_score,
        keyword_score=keyword_score,
    )


def candidate_from_ranked_hit(hit: RankedHit) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        text=hit.text,
        metadata={
            "page_number": hit.page_number,
            "document_title": hit.document_title,
            "section_path": list(hit.section_path),
            "content_type": hit.content_type,
            "token_count": hit.token_count,
        },
        vector_score=hit.similarity,
    )


def ranked_hit_from_candidate(candidate: RetrievalCandidate, *, rank: int) -> RankedHit:
    score = candidate.rerank_score
    if score is None:
        score = candidate.vector_score
    if score is None:
        score = candidate.keyword_score
    return RankedHit(
        rank=rank,
        similarity=float(score or 0.0),
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        section_path=candidate.section_path,
        token_count=candidate.token_count,
        content_type=candidate.content_type,
        text=candidate.text,
        page_number=candidate.page_number,
        document_title=candidate.document_title,
    )


def ranked_hits_from_candidates(candidates: list[RetrievalCandidate]) -> list[RankedHit]:
    return [
        ranked_hit_from_candidate(candidate, rank=index)
        for index, candidate in enumerate(candidates, start=1)
    ]
