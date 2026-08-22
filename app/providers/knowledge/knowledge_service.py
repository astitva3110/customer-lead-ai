from __future__ import annotations

from typing import Any

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import HybridSearchResult, RetrievalCandidate
from app.providers.knowledge.llama_retriever import HybridLlamaRetriever


class LlamaIndexKnowledgeService:
    """RAG/retrieval layer. LlamaIndex on the outside, existing hybrid/PGVector underneath."""

    def __init__(self, hybrid: HybridRetriever, *, retrieval_version: str = "") -> None:
        self._hybrid = hybrid
        self._llama = HybridLlamaRetriever(hybrid)
        self._retrieval_version = retrieval_version

    def retrieve_knowledge(self, query: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
        del context
        nodes = list(self._llama.retrieve(query))
        detailed = self._llama.last_result
        preview = None if detailed is None else _preview_search(detailed, None)
        chunks = []
        scores = []
        metadata = []
        for node in nodes:
            meta = dict(node.node.metadata or {})
            chunks.append(
                {
                    "chunk_id": meta.get("chunk_id") or node.node.node_id,
                    "document_id": meta.get("document_id") or "",
                    "section_path": meta.get("section_path") or [],
                    "page_number": meta.get("page_number"),
                    "title": meta.get("title") or "",
                    "content_type": meta.get("content_type") or "paragraph",
                    "text": node.node.get_content(),
                    "score": float(node.score or 0.0),
                    "knowledge_key": meta.get("knowledge_key"),
                    "chunking_algorithm_version": meta.get("chunking_algorithm_version"),
                    "embedding_input_manifest": meta.get("embedding_input_manifest"),
                }
            )
            scores.append(float(node.score or 0.0))
            metadata.append(meta)
        return {
            "chunks": chunks,
            "scores": scores,
            "metadata": metadata,
            "query": query,
            "retrieval_version": self._retrieval_version,
            "vector_candidate_count": 0 if detailed is None else len(detailed.vector_candidates),
            "keyword_candidate_count": 0 if detailed is None else len(detailed.keyword_candidates),
            "merged_candidate_count": 0 if detailed is None else len(detailed.merged_candidates),
            "reranked_count": 0 if detailed is None else len(detailed.reranked_candidates),
            "timings": None if detailed is None else detailed.timings.to_dict(),
            "retrieval_preview": preview,
            "backend": "llamaindex-hybrid",
            "reranker_name": type(getattr(self._hybrid, "_reranker", None)).__name__,
            "reranker_model": getattr(settings, "reranker_model", None),
            "retrieval_k": getattr(self._hybrid, "_final_k", None),
            "vector_k": getattr(self._hybrid, "_vector_k", None),
            "keyword_k": getattr(self._hybrid, "_keyword_k", None),
            "vector_table": getattr(settings, "phase12_vector_table", None),
            "embedding_model": getattr(settings, "embedding_model", None),
            "embedding_dimension": getattr(settings, "embedding_dimension", None),
            "llamaindex_service": "LlamaIndexKnowledgeService",
            "retriever_used": "HybridLlamaRetriever",
        }


def _preview_search(detailed: HybridSearchResult, top_k: int | None) -> dict[str, list[dict[str, Any]]]:
    return {
        "vector": [_preview_candidate(item, "vector") for item in _take(detailed.vector_candidates, top_k)],
        "keyword": [_preview_candidate(item, "keyword") for item in _take(detailed.keyword_candidates, top_k)],
        "merged": [_preview_candidate(item, "merged") for item in _take(detailed.merged_candidates, top_k)],
        "reranked": [_preview_candidate(item, "rerank") for item in _take(detailed.reranked_candidates, top_k)],
        "final": [_preview_candidate(item, "final") for item in _take(detailed.final_candidates, top_k)],
    }


def _take(items: list[RetrievalCandidate], top_k: int | None) -> list[RetrievalCandidate]:
    if top_k is None:
        return list(items)
    return list(items[: max(1, top_k)])


def _preview_text(text: str | None) -> str:
    value = text or ""
    limit = max(0, int(settings.chat_trace_text_preview_chars or 0))
    if limit and len(value) > limit:
        return value[:limit] + "..."
    return value


def _round_score(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _preview_candidate(item: RetrievalCandidate, kind: str) -> dict[str, Any]:
    if kind == "vector":
        score = item.vector_score
    elif kind == "keyword":
        score = item.keyword_score
    elif kind == "rerank":
        score = item.rerank_score
    else:
        score = item.rerank_score if item.rerank_score is not None else item.vector_score
    return {
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "score": _round_score(score),
        "vector_score": _round_score(item.vector_score),
        "keyword_score": _round_score(item.keyword_score),
        "combined_score": None,
        "rerank_score": _round_score(item.rerank_score),
        "original_retrieval_score": _round_score(item.vector_score),
        "section": " > ".join(item.section_path),
        "title": item.document_title,
        "token_count": item.token_count,
        "chunk_type": item.content_type,
        "text": _preview_text(item.text),
    }
