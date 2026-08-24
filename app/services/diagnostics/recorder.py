from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.services.diagnostics.ids import new_trace_id
from app.services.diagnostics.models import ChatTrace
from app.services.diagnostics.redact import redact_mapping, redact_text
from app.helpers.query_normalize import canonicalize_knowledge_query

_current: ContextVar[ChatTrace | None] = ContextVar("chat_trace", default=None)


def current_trace() -> ChatTrace | None:
    return _current.get()


def tracing_enabled() -> bool:
    return bool(getattr(settings, "chat_trace_enabled", False) or getattr(settings, "chat_debug_console", False))


def _preview(text: str | None) -> str:
    limit = max(0, int(settings.chat_trace_text_preview_chars or 0))
    value = redact_text(text or "")
    if limit and len(value) > limit:
        return value[:limit] + "..."
    return value


def _full_or_preview(text: str | None) -> str:
    if settings.chat_trace_include_full_context or settings.chat_debug_console:
        return redact_text(text or "")
    return _preview(text)


def diagnostic_normalized_query(message: str) -> str:
    return canonicalize_knowledge_query(message or "")


class TraceSession:
    def __init__(self, trace: ChatTrace, token: Token) -> None:
        self.trace = trace
        self._token = token

    @classmethod
    def start(cls, state: Any, message: str) -> TraceSession:
        trace = ChatTrace()
        trace.request = {
            "conversation_id": state.conversation_id,
            "trace_id": new_trace_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": redact_text(message),
        }
        trace.state_before = {
            "conversation_goal": str(state.conversation_goal or ""),
            "conversation_status": _conversation_status(state),
            "current_turn_intent": str(state.current_turn_intent or ""),
            "current_status": str(state.current_turn_intent or ""),
            "current_product": state.product or None,
            "lead_stage": str(state.lead_stage or ""),
            "user_context_keys": sorted((state.user_context or {}).keys()),
            "lead_workflow": str(state.lead_workflow or ""),
            "support_workflow": str(state.support_workflow or ""),
            "awaiting_field": state.awaiting_field or None,
            "previous_assistant_message": _preview(_previous_assistant_message(state)),
            "history_turn_count": len(state.conversation_history or []),
        }
        token = _current.set(trace)
        return cls(trace, token)

    def finish(self, state: Any) -> None:
        meta = dict(state.retrieval_metadata or {})
        flags = dict(state.trace or {})
        timings = dict(flags.get("retrieval_stage_ms") or {})
        self.trace.response = {
            "final_response": redact_text(state.response or ""),
            "final_mode": str(state.mode or ""),
            "sources_count": len(state.sources or []),
            "source_ids": [item.get("chunk_id") for item in (state.sources or []) if item.get("chunk_id")],
        }
        self.trace.state_after = {
            "conversation_goal": str(state.conversation_goal or ""),
            "conversation_status": _conversation_status(state),
            "current_turn_intent": str(state.current_turn_intent or ""),
            "current_status": flags.get("current_status") or str(state.current_turn_intent or ""),
            "current_product": state.product or None,
            "active_product": flags.get("active_product") or state.product or None,
            "diverge": flags.get("diverge"),
            "sub_questions": flags.get("sub_questions") or [],
            "lead_stage": str(state.lead_stage or flags.get("lead_stage") or ""),
            "lead_status": str(state.lead_stage or flags.get("lead_status") or ""),
            "lead_collection_active": bool(getattr(state, "lead_collection_active", False)),
        "next_action": flags.get("next_action"),
            "resolved_query": flags.get("resolved_query"),
            "history_turn_count": len(state.conversation_history or []),
            "lead_status_db": str(state.lead_status or ""),
            "support_status": str(state.ticket_status or ""),
            "lead_state": str(state.lead_workflow or ""),
            "support_state": str(state.support_workflow or ""),
            "awaiting_field": state.awaiting_field or None,
            "user_context_keys": sorted((state.user_context or {}).keys()),
            "user_context": redact_mapping(state.user_context or {}),
        }
        tool = flags.get("tool_called") or ""
        failed = state.error in {"lead_create_failed", "ticket_create_failed"}
        if tool or failed:
            self.trace.tool_execution = {
                "tool_executed": bool(tool),
                "tool_name": tool or None,
                "requested": bool(tool) or failed,
                "executed": bool(tool) or failed,
                "success": bool(tool) and not failed,
                "failure_reason": state.error if failed else None,
                "latency_ms": flags.get("tool_ms"),
                "lead_id": flags.get("lead_id"),
                "ticket_id": flags.get("ticket_id"),
            }
        else:
            self.trace.tool_execution = {
                "tool_executed": False,
                "lead_id": flags.get("lead_id"),
                "ticket_id": flags.get("ticket_id"),
            }
        self.trace.latency = {
            "guardrail_ms": (self.trace.guardrail or {}).get("latency_ms"),
            "router_ms": flags.get("routing_ms"),
            "routing_ms": flags.get("routing_ms"),
            "rewrite_ms": flags.get("query_rewrite_ms") if (self.trace.query or {}).get("rewrite_executed") else (
                flags.get("query_rewrite_ms") if flags.get("query_rewrite_ms") is not None else None
            ),
            "retrieval_ms": flags.get("retrieval_ms"),
            "generation_ms": flags.get("generation_ms"),
            "tool_ms": flags.get("tool_ms"),
            "total_ms": flags.get("total_latency_ms"),
            "vector_ms": timings.get("vector_ms"),
            "keyword_ms": timings.get("keyword_ms"),
            "merge_ms": timings.get("merge_ms"),
            "rerank_ms": timings.get("rerank_ms"),
            "threshold_ms": timings.get("threshold_ms"),
        }
        if not self.trace.query:
            original = redact_text(state.user_message or "")
            rewritten = state.query_rewritten or ""
            prepared = diagnostic_normalized_query(state.user_message or "")
            reason = None
            if rewritten:
                reason = "pronoun_product" if rewritten != prepared else "conversational_normalize"
            self.trace.query = {
                "original_user_query": original,
                "normalized_query": prepared,
                "rewritten_query": rewritten or None,
                "rewrite_executed": bool(rewritten),
                "rewrite_reason": reason,
                "rewrite_latency_ms": flags.get("query_rewrite_ms"),
            }
        if not self.trace.turn_understanding:
            understanding = dict(flags.get("turn_understanding") or {})
            method = "llm" if flags.get("turn_understanding_llm") else "deterministic"
            if flags.get("guardrail"):
                method = "fallback"
            self.trace.turn_understanding = {
                "turn_intent": understanding.get("turn_intent") or state.current_turn_intent,
                "needs_rag": understanding.get("needs_rag", flags.get("should_retrieve")),
                "lead_intent": understanding.get("lead_intent", state.lead_intent),
                "support_intent": understanding.get("support_intent", state.support_intent),
                "explicit_action": understanding.get("explicit_action") or state.explicit_action or None,
                "confidence": understanding.get("confidence"),
                "information_updates": redact_mapping(understanding.get("information_updates") or {}),
                "user_context_updates": redact_mapping(understanding.get("user_context_updates") or {}),
                "user_context": redact_mapping(state.user_context or {}),
                "method": method,
            }
        else:
            self.trace.turn_understanding["user_context"] = redact_mapping(state.user_context or {})
        if meta.get("retrieval_version") and not self.trace.retrieval.get("corpus_version"):
            self.trace.retrieval.setdefault("corpus_version", meta.get("retrieval_version"))
        del meta

    def close(self) -> None:
        _current.reset(self._token)


def record_guardrail(*, executed: bool, result: str | None, latency_ms: float) -> None:
    trace = current_trace()
    if not trace:
        return
    trace.guardrail = {
        "executed": executed,
        "result": result or "allow",
        "blocked": bool(result),
        "reason": result,
        "latency_ms": round(latency_ms, 3),
    }


def record_turn_understanding(payload: dict[str, Any], *, method: str) -> None:
    trace = current_trace()
    if not trace:
        return
    trace.turn_understanding = {
        "turn_intent": payload.get("turn_intent"),
        "needs_rag": payload.get("needs_rag"),
        "lead_intent": payload.get("lead_intent"),
        "support_intent": payload.get("support_intent"),
        "explicit_action": payload.get("explicit_action"),
        "confidence": payload.get("confidence"),
        "information_updates": redact_mapping(payload.get("information_updates") or {}),
        "user_context_updates": redact_mapping(payload.get("user_context_updates") or {}),
        "user_context": redact_mapping(payload.get("user_context") or payload.get("user_context_updates") or {}),
        "method": method,
    }


def record_query(state: Any, *, latency_ms: float) -> None:
    trace = current_trace()
    if not trace:
        return
    original = state.user_message or ""
    rewritten = state.query_rewritten or ""
    prepared = diagnostic_normalized_query(original)
    reason = None
    if rewritten:
        reason = "pronoun_product" if rewritten != prepared else "conversational_normalize"
    trace.query = {
        "original_user_query": redact_text(original),
        "normalized_query": diagnostic_normalized_query(original),
        "rewritten_query": rewritten or None,
        "rewrite_executed": bool(rewritten),
        "rewrite_reason": reason,
        "rewrite_latency_ms": round(latency_ms, 3),
    }


def record_retrieval(retrieval: dict[str, Any], *, latency_ms: float) -> None:
    trace = current_trace()
    if not trace:
        return
    limit = _candidate_limit()
    preview = dict(retrieval.get("retrieval_preview") or {})
    vector = list(preview.get("vector") or [])
    keyword = list(preview.get("keyword") or [])
    merged = list(preview.get("merged") or [])
    fallback = _limit_rows(_chunks_as_candidates(retrieval.get("chunks") or []), limit)
    raw = merged or vector or fallback
    raw_rows = [_clip_candidate(item, index) for index, item in enumerate(_limit_rows(raw, limit), start=1)]
    scores = [item.get("score") for item in raw_rows if isinstance(item.get("score"), (int, float))]
    original_rank = {
        str(item.get("chunk_id")): index
        for index, item in enumerate(raw, start=1)
        if item.get("chunk_id")
    }
    timings = dict(retrieval.get("timings") or {})
    trace.retrieval = {
        "backend": retrieval.get("backend") or "hybrid",
        "corpus_version": retrieval.get("retrieval_version") or retrieval.get("corpus_version") or "",
        "vector_table": retrieval.get("vector_table"),
        "embedding_model": retrieval.get("embedding_model") or getattr(settings, "embedding_model", None),
        "embedding_dimension": retrieval.get("embedding_dimension") or getattr(settings, "embedding_dimension", None),
        "retrieval_query": retrieval.get("query"),
        "retrieval_k": retrieval.get("retrieval_k") or retrieval.get("final_k"),
        "vector_k": retrieval.get("vector_k"),
        "keyword_k": retrieval.get("keyword_k"),
        "candidate_count": retrieval.get("merged_candidate_count")
        or retrieval.get("vector_candidate_count")
        or len(raw_rows),
        "vector_candidate_count": retrieval.get("vector_candidate_count"),
        "keyword_candidate_count": retrieval.get("keyword_candidate_count"),
        "merged_candidate_count": retrieval.get("merged_candidate_count"),
        "latency_ms": round(latency_ms, 3),
        "timings": timings,
        "candidates": raw_rows,
        "vector": [_hybrid_row(item, index) for index, item in enumerate(vector, start=1)],
        "keyword": [_hybrid_row(item, index) for index, item in enumerate(keyword, start=1)],
        "merged": [_hybrid_row(item, index) for index, item in enumerate(merged, start=1)],
        "raw_count": len(raw_rows),
        "raw_top_score": max(scores) if scores else None,
        "raw_score_min": min(scores) if scores else None,
        "raw_score_max": max(scores) if scores else None,
    }
    backend = str(retrieval.get("backend") or "")
    trace.llamaindex = {
        "enabled": backend.startswith("llamaindex"),
        "adapter": retrieval.get("retriever_used") or "HybridLlamaRetriever",
        "service": retrieval.get("llamaindex_service") or "LlamaIndexKnowledgeService",
        "retriever": retrieval.get("retriever_used") or "HybridLlamaRetriever",
        "retrieval_query": retrieval.get("query"),
        "retrieval_top_k": retrieval.get("retrieval_k") or retrieval.get("final_k"),
        "embedding_model": retrieval.get("embedding_model") or getattr(settings, "embedding_model", None),
        "embedding_dimension": retrieval.get("embedding_dimension") or getattr(settings, "embedding_dimension", None),
        "nodes": len(retrieval.get("chunks") or []),
        "detailed_result": bool(preview or retrieval.get("timings")),
    }
    reranked = list(preview.get("reranked") or [])
    rerank_rows = []
    for index, item in enumerate(reranked, start=1):
        row = _clip_candidate(item, index)
        chunk_id = str(item.get("chunk_id") or "")
        row["original_retrieval_rank"] = original_rank.get(chunk_id)
        rerank_rows.append(row)
    if reranked or timings.get("rerank_ms") is not None:
        trace.reranking = {
            "enabled": True,
            "name": retrieval.get("reranker_name"),
            "model": retrieval.get("reranker_model") or getattr(settings, "reranker_model", None),
            "input_count": retrieval.get("merged_candidate_count") or len(merged) or None,
            "output_count": retrieval.get("reranked_count") or len(rerank_rows),
            "latency_ms": timings.get("rerank_ms"),
            "input": [_hybrid_row(item, index) for index, item in enumerate(merged or raw, start=1)],
            "candidates": rerank_rows,
        }
    else:
        trace.reranking = {"enabled": False}


def record_final_context(hits: list[Any]) -> None:
    trace = current_trace()
    if not trace:
        return
    chunks = []
    for index, hit in enumerate(hits, start=1):
        text = getattr(hit, "text", None)
        if text is None and isinstance(hit, dict):
            text = hit.get("text")
        chunks.append(
            {
                "position": index,
                "chunk_id": getattr(hit, "chunk_id", None) or (hit.get("chunk_id") if isinstance(hit, dict) else ""),
                "document_id": getattr(hit, "document_id", None) or (hit.get("document_id") if isinstance(hit, dict) else ""),
                "title": getattr(hit, "document_title", None) or (hit.get("title") if isinstance(hit, dict) else ""),
                "section_path": list(getattr(hit, "section_path", None) or (hit.get("section_path") if isinstance(hit, dict) else []) or []),
                "score": getattr(hit, "similarity", None) if not isinstance(hit, dict) else hit.get("score"),
                "text": _full_or_preview(text),
            }
        )
    trace.final_context = {"context_count": len(chunks), "chunks": chunks}


def record_generation(
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    latency_ms: float,
    llm_call_count: int,
    result: dict[str, Any] | None = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    raw_output: str | None = None,
    parsed_payload: dict[str, Any] | None = None,
    explained: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> None:
    trace = current_trace()
    if not trace:
        return
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model or getattr(settings, "generation_model", ""),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "input_token_count": None,
        "output_token_count": None,
        "latency_ms": round(latency_ms, 3),
        "llm_call_count": llm_call_count,
    }
    if result:
        payload["generation_result"] = {
            "grounded": result.get("grounded"),
            "answer": redact_text(str(result.get("answer") or "")),
            "source_ids": list(result.get("source_ids") or []),
        }
    if raw_output is not None:
        payload["raw_output"] = redact_text(raw_output)
        payload["raw_model_output"] = redact_text(raw_output)
    _attach_parsed_generation(payload, parsed_payload, explained)
    user_text = user_prompt if user_prompt is not None else prompt
    if _prompt_capture_allowed():
        if system_prompt:
            payload["system_prompt"] = _prompt_text(system_prompt)
        if user_text:
            payload["user_prompt"] = _prompt_text(user_text)
            payload["prompt"] = _prompt_text(user_text)
            knowledge, remainder = _split_knowledge_context(user_text)
            if knowledge:
                payload["knowledge_context"] = _prompt_text(knowledge)
            if remainder:
                payload["user_prompt_body"] = _prompt_text(remainder)
    trace.generation = payload
    if error:
        record_error("generation", error, recoverable=True, fallback_used=True)


def _attach_parsed_generation(
    payload: dict[str, Any],
    parsed_payload: dict[str, Any] | None,
    explained: dict[str, Any] | None,
) -> None:
    if isinstance(parsed_payload, dict):
        payload["parsed_grounded"] = parsed_payload.get("grounded")
        if "answer" in parsed_payload:
            payload["parsed_answer"] = redact_text(str(parsed_payload.get("answer") or ""))
        else:
            payload["parsed_answer"] = None
        payload["parsed_source_ids"] = parsed_payload.get("source_ids")
    else:
        payload["parsed_grounded"] = None
        payload["parsed_answer"] = None
        payload["parsed_source_ids"] = None
    if explained:
        payload["valid_source_ids"] = list(explained.get("valid_source_ids") or [])
        payload["invalid_source_ids"] = list(explained.get("invalid_source_ids") or [])
        payload["validator_reason"] = explained.get("validator_reason")


def record_grounding(payload: dict[str, Any]) -> None:
    trace = current_trace()
    if not trace:
        return
    trace.grounding = payload


def record_error(component: str, error: Exception, *, recoverable: bool, fallback_used: bool) -> None:
    trace = current_trace()
    if not trace:
        return
    message = redact_text(str(error))
    lowered = message.lower()
    for secret in ("api_key", "authorization", "bearer ", "password"):
        if secret in lowered:
            message = "[redacted-error]"
            break
    trace.errors.append(
        {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": message,
            "recoverable": recoverable,
            "fallback_used": fallback_used,
        }
    )


def _prompt_capture_allowed() -> bool:
    return bool(tracing_enabled() and (settings.chat_trace_include_prompt or settings.chat_debug_console))


def _debug_full_text() -> bool:
    return bool(settings.chat_debug_console or settings.chat_trace_include_full_context)


def _prompt_text(text: str | None) -> str:
    if _debug_full_text():
        return redact_text(text or "")
    return _preview(text)


def _candidate_limit() -> int | None:
    if _debug_full_text():
        return None
    return max(1, int(settings.chat_trace_retrieval_top_k or 10))


def _limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    items = list(rows or [])
    if limit is None:
        return items
    return items[:limit]


def _split_knowledge_context(user_prompt: str) -> tuple[str, str]:
    text = user_prompt or ""
    marker = "Knowledge-base context:"
    question = "User question:"
    if marker not in text:
        return "", text
    after = text.split(marker, 1)[1]
    if question in after:
        knowledge, remainder = after.split(question, 1)
        return knowledge.strip(), f"{question}{remainder}".strip()
    return after.strip(), ""


def _conversation_status(state: Any) -> str:
    goal = str(getattr(state, "conversation_goal", "") or "")
    if goal in {"LEAD", "SALES"}:
        return "SALE"
    if goal == "SUPPORT":
        return "SUPPORT"
    return "NONE"


def _previous_assistant_message(state: Any) -> str:
    history = list(getattr(state, "conversation_history", None) or [])
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return str(item.get("content") or "")
    return ""


def _chunks_as_candidates(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in chunks:
        section = item.get("section_path") or []
        rows.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "score": item.get("score"),
                "title": item.get("title"),
                "section": " > ".join(str(part) for part in section),
                "token_count": item.get("token_count") or max(1, len(str(item.get("text") or "").split())),
                "chunk_type": item.get("content_type") or item.get("chunk_type"),
                "text": item.get("text"),
            }
        )
    return rows


def _clip_candidate(item: dict[str, Any], rank: int) -> dict[str, Any]:
    section = item.get("section") or item.get("section_path")
    return {
        "rank": rank,
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "score": item.get("score") if item.get("score") is not None else item.get("rerank_score"),
        "title": item.get("title"),
        "section_path": section,
        "token_count": item.get("token_count"),
        "chunk_type": item.get("chunk_type") or item.get("content_type"),
        "text_preview": _preview(item.get("text") or item.get("text_preview")),
        "rerank_score": item.get("rerank_score"),
        "original_retrieval_score": item.get("original_retrieval_score") or item.get("vector_score"),
        "original_retrieval_rank": item.get("original_retrieval_rank"),
        "vector_score": item.get("vector_score"),
        "keyword_score": item.get("keyword_score"),
        "combined_score": item.get("combined_score"),
    }


def _hybrid_row(item: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "vector_score": item.get("vector_score"),
        "keyword_score": item.get("keyword_score") or item.get("lexical_score"),
        "combined_score": item.get("combined_score"),
        "score": item.get("score"),
        "section_path": item.get("section") or item.get("section_path"),
        "title": item.get("title"),
        "text_preview": _preview(item.get("text") or item.get("text_preview")),
        "rerank_score": item.get("rerank_score"),
        "original_retrieval_rank": item.get("original_retrieval_rank"),
    }
