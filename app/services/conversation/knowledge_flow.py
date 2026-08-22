from __future__ import annotations

import time
from typing import Any

from app.services.conversation.models import ConversationState
from app.services.conversation.query_rewriter import QueryRewriter
from app.services.diagnostics.recorder import current_trace, record_final_context, record_query, record_retrieval
from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.interfaces.providers.knowledge import KnowledgeService
from app.kb.evaluation.models import RankedHit


class KnowledgeFlow:
    """Knowledge mode: optional rewrite → LlamaIndex RAG → existing generation guardrail."""

    def __init__(
        self,
        knowledge: KnowledgeService,
        generation: GenerationService,
        rewriter: QueryRewriter | None = None,
        *,
        model_used: str = "",
    ) -> None:
        self._knowledge = knowledge
        self._generation = generation
        self._rewriter = rewriter or QueryRewriter()
        self._model_used = model_used

    def handle(self, state: ConversationState) -> ConversationState:
        if not state.trace.get("should_retrieve", True):
            state.response = state.response or "Thanks, I've noted that."
            state.sources = []
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            return state
        rewrite_started = time.perf_counter()
        original_query = state.user_message or ""
        self._rewriter.apply(state)
        retrieval_query = state.query_rewritten or original_query
        rewrite_ms = round((time.perf_counter() - rewrite_started) * 1000, 3)
        state.trace["query_original"] = original_query
        state.trace["query_rewrite_ms"] = rewrite_ms
        if current_trace():
            record_query(state, latency_ms=rewrite_ms)
        retrieval_started = time.perf_counter()
        retrieval = self._knowledge.retrieve_knowledge(retrieval_query, context=state.to_dict())
        state.trace["retrieval_started"] = True
        state.trace["retrieval_used"] = True
        state.trace["retrieval_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 3)
        chunks = retrieval.get("chunks") or []
        state.retrieved_context = chunks
        state.retrieved_chunk_ids = [item.get("chunk_id", "") for item in chunks]
        state.retrieval_metadata = {
            "query": retrieval.get("query", retrieval_query),
            "retrieval_version": retrieval.get("retrieval_version", ""),
            "vector_candidate_count": retrieval.get("vector_candidate_count", 0),
            "keyword_candidate_count": retrieval.get("keyword_candidate_count", 0),
            "merged_candidate_count": retrieval.get("merged_candidate_count", 0),
            "reranked_count": retrieval.get("reranked_count", 0),
            "final_context_count": len(chunks),
            "query_rewritten": bool(state.query_rewritten),
        }
        if retrieval.get("timings"):
            state.trace["retrieval_stage_ms"] = retrieval["timings"]
        if current_trace():
            record_retrieval(retrieval, latency_ms=float(state.trace["retrieval_ms"]))
            preview = retrieval.get("retrieval_preview")
            if preview:
                state.trace["retrieval_preview"] = preview
            else:
                top_k = 10
                try:
                    from app.config import settings

                    top_k = max(1, int(settings.chat_trace_retrieval_top_k or 10))
                except Exception:
                    pass
                state.trace["retrieval_preview"] = {
                    "final": [_chunk_preview(item) for item in chunks[:top_k]],
                }
        hits = _hits_from_chunks(chunks)
        if not hits:
            state.response = INSUFFICIENT_INFORMATION_MESSAGE
            state.sources = []
            state.trace["model_used"] = ""
            state.trace["retrieval_used"] = False
            return state
        if current_trace():
            record_final_context(hits)
        generation_started = time.perf_counter()
        result = self._generation.generate(original_query, hits)
        trace = current_trace()
        if trace and trace.generation:
            trace.generation["model"] = self._model_used
        state.trace["generation_ms"] = round((time.perf_counter() - generation_started) * 1000, 3)
        state.trace["generation_temperature"] = self._generation.temperature
        state.trace["llm_call_count"] = int(state.trace.get("llm_call_count") or 0) + 1
        state.trace["model_used"] = self._model_used
        state.trace["grounded"] = bool(result.grounded)
        state.trace["source_ids"] = list(result.source_ids)
        state.response = result.answer
        if not result.grounded:
            state.sources = []
            return state
        by_id = {hit.chunk_id: hit for hit in hits}
        state.sources = [
            {
                "chunk_id": source_id,
                "url": f"document://{by_id[source_id].document_id}",
                "title": by_id[source_id].document_title,
                "score": by_id[source_id].similarity,
            }
            for source_id in result.source_ids
            if source_id in by_id
        ]
        return state


def _chunk_preview(item: dict[str, Any]) -> dict[str, Any]:
    section = item.get("section_path") or []
    return {
        "chunk_id": str(item.get("chunk_id") or ""),
        "score": None if item.get("score") is None else round(float(item.get("score") or 0.0), 4),
        "section": " > ".join(str(part) for part in section),
        "title": str(item.get("title") or ""),
        "text": str(item.get("text") or "")[:180],
    }


def _hits_from_chunks(chunks: list[dict[str, Any]]) -> list[RankedHit]:
    hits: list[RankedHit] = []
    for index, item in enumerate(chunks, start=1):
        hits.append(
            RankedHit(
                rank=index,
                similarity=float(item.get("score") or 0.0),
                chunk_id=str(item.get("chunk_id") or ""),
                document_id=str(item.get("document_id") or ""),
                section_path=list(item.get("section_path") or []),
                token_count=max(1, len(str(item.get("text") or "").split())),
                content_type=str(item.get("content_type") or "paragraph"),
                text=str(item.get("text") or ""),
                page_number=item.get("page_number"),
                document_title=str(item.get("title") or ""),
            )
        )
    return hits
