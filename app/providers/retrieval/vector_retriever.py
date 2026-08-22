from __future__ import annotations

from app.services.retrieval.models import RetrievalCandidate
from app.helpers.retrieval_hits import candidate_from_ranked_hit, candidate_from_store_item
from app.kb.evaluation.models import RankedHit
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore
from app.kb.retrieval.service import RetrievalService


class VectorCandidateRetriever:
    """Existing pgvector cosine search, exposed as a candidate retriever."""

    def __init__(
        self,
        *,
        store: Phase12VectorStore | None = None,
        session: EmbeddingSession | None = None,
        embedding_version: str | None = None,
        search_service: RetrievalService | None = None,
        default_k: int = 20,
    ) -> None:
        if search_service is None and (store is None or session is None):
            raise ValueError("VectorCandidateRetriever requires search_service or store+session")
        self._store = store
        self._session = session
        self._embedding_version = embedding_version
        self._search_service = search_service
        self._default_k = default_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        limit = top_k if top_k > 0 else self._default_k
        if self._store is not None and self._session is not None:
            query_vector = self._session.embed_queries([query])[0]
            raw = self._store.search(
                query_vector,
                top_k=limit,
                embedding_version=self._embedding_version,
                document_id=document_id,
            )
            return [
                candidate_from_store_item(item, vector_score=float(item.get("similarity") or 0.0))
                for item in raw
            ]

        hits: list[RankedHit] = self._search_service.search(query, top_k=limit)
        if document_id:
            hits = [hit for hit in hits if hit.document_id == document_id]
        return [candidate_from_ranked_hit(hit) for hit in hits]
