from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class RetrievalCandidate:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None

    @property
    def page_number(self) -> int | None:
        return self.metadata.get("page_number")

    @property
    def document_title(self) -> str:
        return str(self.metadata.get("document_title") or "")

    @property
    def section_path(self) -> list[str]:
        path = self.metadata.get("section_path") or []
        return list(path)

    @property
    def content_type(self) -> str:
        return str(self.metadata.get("content_type") or "paragraph")

    @property
    def token_count(self) -> int:
        raw = self.metadata.get("token_count")
        if isinstance(raw, int):
            return raw
        return max(1, len(self.text.split()))

    def with_scores(
        self,
        *,
        vector_score: float | None = None,
        keyword_score: float | None = None,
        rerank_score: float | None = None,
    ) -> RetrievalCandidate:
        return replace(
            self,
            vector_score=self.vector_score if vector_score is None else vector_score,
            keyword_score=self.keyword_score if keyword_score is None else keyword_score,
            rerank_score=self.rerank_score if rerank_score is None else rerank_score,
        )


RankedCandidate = RetrievalCandidate


@dataclass
class StageTimings:
    vector_ms: float = 0.0
    keyword_ms: float = 0.0
    merge_ms: float = 0.0
    rerank_ms: float = 0.0
    threshold_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "vector_ms": round(self.vector_ms, 3),
            "keyword_ms": round(self.keyword_ms, 3),
            "merge_ms": round(self.merge_ms, 3),
            "rerank_ms": round(self.rerank_ms, 3),
            "threshold_ms": round(self.threshold_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class HybridSearchResult:
    query: str
    vector_candidates: list[RetrievalCandidate]
    keyword_candidates: list[RetrievalCandidate]
    merged_candidates: list[RetrievalCandidate]
    reranked_candidates: list[RankedCandidate]
    final_candidates: list[RankedCandidate]
    min_score: float
    has_relevant_context: bool
    timings: StageTimings = field(default_factory=StageTimings)

    @property
    def results(self) -> list[RankedCandidate]:
        return self.final_candidates
