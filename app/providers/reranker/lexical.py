from __future__ import annotations

from dataclasses import replace

from app.services.retrieval.models import RankedCandidate, RetrievalCandidate
from app.helpers.lexical_overlap import (
    default_idf_value,
    document_idf,
    idf_weighted_coverage_score,
)


class LexicalOverlapReranker:
    """Query-chunk term coverage reranker. No extra model download."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RankedCandidate]:
        haystacks = [_candidate_haystack(item) for item in candidates]
        idf = document_idf(haystacks)
        unseen_idf = default_idf_value(len(haystacks))
        scored: list[RankedCandidate] = []
        for candidate, haystack in zip(candidates, haystacks):
            score = idf_weighted_coverage_score(query, haystack, idf, default_idf=unseen_idf)
            scored.append(replace(candidate, rerank_score=round(score, 6)))
        scored.sort(
            key=lambda item: (
                item.rerank_score or 0.0,
                item.vector_score or 0.0,
                item.keyword_score or 0.0,
            ),
            reverse=True,
        )
        return scored


def _candidate_haystack(candidate: RetrievalCandidate) -> str:
    return " ".join(
        [
            candidate.text,
            candidate.document_title,
            " ".join(candidate.section_path),
        ]
    )
