from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.providers.reranker.factory import create_reranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore

DOC_ID = "6bb9a4ee-ecf4-58ff-9eaa-3206d27d9b0d"
CONFIG_PATH = Path("configs/retrieval/v2.yaml")


@pytest.fixture(scope="module")
def production_hybrid():
    config = RetrievalConfig.from_yaml(CONFIG_PATH)
    store = Phase12VectorStore(table_name=config.vector_table)
    try:
        probe = store.search_keyword("TINY", top_k=1, document_id=DOC_ID)
    except Exception as exc:
        pytest.skip(f"PostgreSQL/pgvector unavailable: {exc}")
    if not probe and not store.search_keyword("Earkart", top_k=1):
        pytest.skip("Indexed Phase 12 corpus is not available; not re-ingesting")
    session = EmbeddingSession()
    vector = VectorCandidateRetriever(
        store=store,
        session=session,
        embedding_version=config.embedding_version,
        default_k=settings.knowledge_vector_top_k,
    )
    keyword = KeywordCandidateRetriever(
        store,
        embedding_version=config.embedding_version,
        default_k=settings.knowledge_keyword_top_k,
        ensure_index=False,
    )
    hybrid = HybridRetriever(
        vector,
        keyword,
        create_reranker(settings.reranker_provider),
        vector_k=settings.knowledge_vector_top_k,
        keyword_k=settings.knowledge_keyword_top_k,
        final_k=settings.knowledge_final_context_k,
        min_score=settings.reranker_min_score,
        rerank_candidate_k=settings.knowledge_reranker_candidate_k,
    )
    return {
        "config": config,
        "store": store,
        "hybrid": hybrid,
        "vector": vector,
        "keyword": keyword,
        "session": session,
    }
