from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import HybridSearchResult, RetrievalCandidate
from app.providers.knowledge.knowledge_service import _preview_search


class HybridKnowledgeService:
    """Native RAG retrieval via the existing HybridRetriever (no LlamaIndex)."""

    def __init__(self, hybrid: HybridRetriever, *, retrieval_version: str = "") -> None:
        self._hybrid = hybrid
        self._retrieval_version = retrieval_version
        self._last_result: HybridSearchResult | None = None

    def retrieve_knowledge(self, query: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        del context
        detailed = self._hybrid.search_detailed(query)
        self._last_result = detailed
        preview = _preview_search(detailed, None)
        chunks: list[dict[str, Any]] = []
        scores: list[float] = []
        metadata: list[dict[str, Any]] = []
        for candidate in detailed.final_candidates:
            meta = dict(candidate.metadata or {})
            chunks.append(_chunk_from_candidate(candidate, meta))
            score = candidate.rerank_score if candidate.rerank_score is not None else candidate.vector_score
            scores.append(float(score or 0.0))
            metadata.append(meta)
        return {
            "chunks": chunks,
            "scores": scores,
            "metadata": metadata,
            "query": query,
            "retrieval_version": self._retrieval_version,
            "vector_candidate_count": len(detailed.vector_candidates),
            "keyword_candidate_count": len(detailed.keyword_candidates),
            "merged_candidate_count": len(detailed.merged_candidates),
            "reranked_count": len(detailed.reranked_candidates),
            "timings": detailed.timings.to_dict(),
            "retrieval_preview": preview,
            "backend": "native-hybrid",
            "reranker_name": type(getattr(self._hybrid, "_reranker", None)).__name__,
            "reranker_model": getattr(settings, "reranker_model", None),
            "retrieval_k": getattr(self._hybrid, "_final_k", None),
            "vector_k": getattr(self._hybrid, "_vector_k", None),
            "keyword_k": getattr(self._hybrid, "_keyword_k", None),
            "vector_table": getattr(settings, "phase12_vector_table", None),
            "embedding_model": getattr(settings, "embedding_model", None),
            "embedding_dimension": getattr(settings, "embedding_dimension", None),
            "retriever_used": "HybridRetriever",
        }


def _chunk_from_candidate(candidate: RetrievalCandidate, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "section_path": candidate.section_path,
        "page_number": candidate.page_number,
        "title": candidate.document_title,
        "content_type": candidate.content_type,
        "text": candidate.text,
        "score": float(
            candidate.rerank_score if candidate.rerank_score is not None else (candidate.vector_score or 0.0)
        ),
        "knowledge_key": meta.get("knowledge_key"),
        "chunking_algorithm_version": meta.get("chunking_algorithm_version"),
        "embedding_input_manifest": meta.get("embedding_input_manifest"),
    }
