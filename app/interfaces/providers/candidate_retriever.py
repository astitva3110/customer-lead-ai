from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.retrieval.models import RetrievalCandidate


class CandidateRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]: ...
