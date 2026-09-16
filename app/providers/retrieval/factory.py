from __future__ import annotations

from app.config import settings
from app.interfaces.providers.retriever import Retriever
from app.services.retrieval.hybrid import HybridRetriever
from app.helpers.retrieval_hits import ranked_hits_from_candidates
from app.providers.reranker.factory import create_reranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.retrieval.runtime_config import build_production_vector_store
from app.kb.retrieval.service import RetrievalService


class HybridChatRetriever:
    """Retriever protocol adapter: hybrid core result → RankedHit list for chat."""

    def __init__(self, hybrid: HybridRetriever) -> None:
        self._hybrid = hybrid

    def search_detailed(self, query: str, *, final_k: int | None = None, document_id: str | None = None, min_score: float | None = None):
        return self._hybrid.search_detailed(
            query,
            final_k=final_k,
            document_id=document_id,
            min_score=min_score,
        )

    def search(self, query: str, *, top_k: int | None = None, document_id: str | None = None):
        result = self._hybrid.search_detailed(query, final_k=top_k, document_id=document_id)
        return ranked_hits_from_candidates(result.final_candidates)


def build_hybrid_retriever(
    config: RetrievalConfig,
    *,
    vector_service: RetrievalService | None = None,
    device: str | None = None,
    vector_k: int | None = None,
    keyword_k: int | None = None,
    final_k: int | None = None,
    rerank_candidate_k: int | None = None,
    min_score: float | None = None,
) -> HybridRetriever:
    store = build_production_vector_store(config)
    vk = settings.vector_candidate_k if vector_k is None else vector_k
    kk = settings.keyword_candidate_k if keyword_k is None else keyword_k
    fk = settings.final_retrieval_k if final_k is None else final_k
    threshold = settings.reranker_min_score if min_score is None else min_score
    if vector_service is not None:
        vector_retriever = VectorCandidateRetriever(
            search_service=vector_service,
            default_k=vk,
        )
    else:
        session = EmbeddingSession(device=device)
        vector_retriever = VectorCandidateRetriever(
            store=store,
            session=session,
            embedding_version=config.embedding_version,
            default_k=vk,
        )
    keyword_retriever = KeywordCandidateRetriever(
        store,
        embedding_version=config.embedding_version,
        default_k=kk,
    )
    return HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        reranker=create_reranker(settings.reranker_provider),
        vector_k=vk,
        keyword_k=kk,
        final_k=fk,
        min_score=threshold,
        rerank_candidate_k=rerank_candidate_k,
    )


def build_knowledge_hybrid_retriever(
    config: RetrievalConfig,
    *,
    vector_service: RetrievalService | None = None,
) -> HybridRetriever:
    return build_hybrid_retriever(
        config,
        vector_service=vector_service,
        vector_k=settings.knowledge_vector_top_k,
        keyword_k=settings.knowledge_keyword_top_k,
        final_k=settings.knowledge_final_context_k,
        rerank_candidate_k=settings.knowledge_reranker_candidate_k,
    )


def build_chat_retriever(config: RetrievalConfig) -> Retriever:
    vector_service = RetrievalService.from_config(config)
    if not settings.hybrid_retrieval_enabled:
        return vector_service
    hybrid = build_hybrid_retriever(config, vector_service=vector_service)
    return HybridChatRetriever(hybrid)
