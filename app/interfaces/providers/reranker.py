from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.retrieval.models import RankedCandidate, RetrievalCandidate


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RankedCandidate]: ...
