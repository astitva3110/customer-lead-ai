from __future__ import annotations

import logging
import time
from typing import Any

from app.helpers.conversation_reply import greeting_reply
from app.helpers.bot_guidance import (
    capability_reply,
    insufficient_redirect_reply,
    looks_like_capability_question,
    looks_like_unclear_user_message,
    unclear_redirect_reply,
)
from app.helpers.conversation_turn import is_casual_conversation, is_greeting_only
from app.helpers.user_language import response_language
from app.helpers.workflow_resume import append_workflow_resume_after_knowledge, next_missing_lead_field
from app.helpers.query_normalize import looks_like_price_query
from app.helpers.conversation_reply import price_query_reply
from app.helpers.quick_replies import attach_lead_offer_buttons, attach_product_interest_quick_replies
from app.services.conversation.models import ConversationState
from app.services.conversation.query_rewriter import QueryRewriter
from app.services.diagnostics.recorder import current_trace, record_final_context, record_query, record_retrieval
from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.interfaces.providers.knowledge import KnowledgeService
from app.kb.evaluation.models import RankedHit

logger = logging.getLogger(__name__)


class KnowledgeFlow:
    """Knowledge mode: optional rewrite → native hybrid RAG → existing generation guardrail."""

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
        original_query = state.user_message or ""
        if is_greeting_only(original_query):
            state.response = greeting_reply(original_query, language=response_language(state))
            state.sources = []
            state.trace = dict(state.trace or {})
            state.trace["should_retrieve"] = False
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            state.trace["greeting_shortcut"] = True
            return state
        if is_casual_conversation(original_query):
            from app.helpers.conversation_reply import conversational_fallback

            state.response = conversational_fallback(state)
            state.sources = []
            state.trace = dict(state.trace or {})
            state.trace["should_retrieve"] = False
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            state.trace["social_shortcut"] = True
            return state
        if looks_like_capability_question(original_query):
            state.response = capability_reply(response_language(state))
            state.sources = []
            state.trace = dict(state.trace or {})
            state.trace["should_retrieve"] = False
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            state.trace["bot_guidance_shortcut"] = True
            return state
        if looks_like_unclear_user_message(original_query):
            state.response = unclear_redirect_reply(response_language(state))
            state.sources = []
            state.trace = dict(state.trace or {})
            state.trace["should_retrieve"] = False
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            state.trace["bot_guidance_shortcut"] = True
            return state
        if not state.trace.get("should_retrieve", True):
            state.response = state.response or "Thanks, I've noted that."
            state.sources = []
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            return state
        rewrite_started = time.perf_counter()
        self._rewriter.apply(state)
        trace = dict(state.trace or {})
        sub_questions = [str(item).strip() for item in (trace.get("sub_questions") or []) if str(item).strip()]
        if not sub_questions:
            sub_questions = [state.query_rewritten or original_query]
        retrieval_queries = sub_questions
        retrieval_query = retrieval_queries[0]
        rewrite_ms = round((time.perf_counter() - rewrite_started) * 1000, 3)
        state.trace["query_original"] = original_query
        state.trace["resolved_query"] = retrieval_query
        state.trace["sub_questions"] = retrieval_queries
        state.trace["query_rewrite_ms"] = rewrite_ms
        if looks_like_price_query(original_query):
            reply = price_query_reply(state.product)
            state.response = append_workflow_resume_after_knowledge(state, reply)
            if not state.trace.get("lead_resume_appended"):
                attach_lead_offer_buttons(state)
            state.sources = []
            state.trace["retrieval_used"] = False
            state.trace["retrieval_started"] = False
            state.trace["price_lead_offer"] = True
            state.trace["model_used"] = ""
            logger.info(
                "KNOWLEDGE -> PRICE_SHORTCUT product=%s lead_resume=%s",
                state.product,
                bool(state.trace.get("lead_resume_appended")),
            )
            return state
        if current_trace():
            record_query(state, latency_ms=rewrite_ms)
        retrieval_started = time.perf_counter()
        chunks, retrieval = _retrieve_for_queries(self._knowledge, retrieval_queries, state)
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
        next_missing = next_missing_lead_field(state)
        logger.info(
            "KNOWLEDGE -> RAG intent=%s conversation_goal=%s lead_collection_active=%s "
            "awaiting_field=%s next_missing_field=%s next_action=%s",
            state.intent,
            state.conversation_goal,
            state.lead_collection_active,
            state.awaiting_field,
            next_missing,
            (state.trace or {}).get("next_action"),
        )
        if not hits:
            state.response = append_workflow_resume_after_knowledge(
                state,
                insufficient_redirect_reply(response_language(state)),
            )
            state.sources = []
            state.trace["model_used"] = ""
            state.trace["retrieval_used"] = False
            attach_product_interest_quick_replies(state)
            return state
        if current_trace():
            record_final_context(hits)
        if (state.trace or {}).get("next_action") == "SALES_PITCH_AND_OFFER_CONTACT":
            state.response = ""
            state.sources = []
            state.trace["needs_natural_reply"] = True
            state.trace["sales_pitch_from_rag"] = True
            return state
        generation_started = time.perf_counter()
        result = self._generation.generate(
            original_query,
            hits,
            response_language=response_language(state),
        )
        logger.info("RAG COMPLETE grounded=%s", bool(result.grounded))
        trace = current_trace()
        if trace and trace.generation:
            trace.generation["model"] = self._model_used
        state.trace["generation_ms"] = round((time.perf_counter() - generation_started) * 1000, 3)
        state.trace["generation_temperature"] = self._generation.temperature
        state.trace["llm_call_count"] = int(state.trace.get("llm_call_count") or 0) + 1
        state.trace["model_used"] = self._model_used
        state.trace["grounded"] = bool(result.grounded)
        state.trace["source_ids"] = list(result.source_ids)
        state.response = append_workflow_resume_after_knowledge(state, result.answer)
        if not result.grounded:
            state.sources = []
            attach_product_interest_quick_replies(state)
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
        attach_product_interest_quick_replies(state)
        return state


def _retrieve_for_queries(
    knowledge: KnowledgeService,
    queries: list[str],
    state: ConversationState,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    last_result: dict[str, Any] = {}
    for query in queries:
        result = knowledge.retrieve_knowledge(query, context=state.to_dict())
        last_result = result
        for chunk in result.get("chunks") or []:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                continue
            score = float(chunk.get("score") or 0.0)
            existing = merged.get(chunk_id)
            if existing is None or score > float(existing.get("score") or 0.0):
                merged[chunk_id] = dict(chunk)
    chunks = sorted(merged.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)
    if last_result:
        last_result = dict(last_result)
        last_result["chunks"] = chunks
        last_result["query"] = queries[0] if queries else last_result.get("query", "")
    return chunks, last_result


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
