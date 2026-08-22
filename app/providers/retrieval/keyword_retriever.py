from __future__ import annotations

from app.services.retrieval.models import RetrievalCandidate
from app.helpers.retrieval_hits import candidate_from_store_item
from app.kb.ingestion.indexing import Phase12VectorStore


class KeywordCandidateRetriever:
    """PostgreSQL full-text search over Phase 12 chunk content."""

    def __init__(
        self,
        store: Phase12VectorStore,
        *,
        embedding_version: str | None = None,
        default_k: int = 20,
        ensure_index: bool = True,
    ) -> None:
        self._store = store
        self._embedding_version = embedding_version
        self._default_k = default_k
        if ensure_index:
            self._store.ensure_content_fts_index()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        limit = top_k if top_k > 0 else self._default_k
        raw = self._store.search_keyword(
            query,
            top_k=limit,
            embedding_version=self._embedding_version,
            document_id=document_id,
        )
        return [
            candidate_from_store_item(item, keyword_score=float(item.get("keyword_score") or 0.0))
            for item in raw
        ]
