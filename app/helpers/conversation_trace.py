from __future__ import annotations

import logging
import re
from typing import Any

from app.services.conversation.models import ConversationGoal, ConversationState

logger = logging.getLogger("conversation.trace")

_REDACT_KEYS = frozenset({"phone", "user_message", "user_name"})
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")


def redact_trace(payload: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _REDACT_KEYS and value:
            redacted[key] = "[redacted]"
        elif isinstance(value, dict):
            redacted[key] = redact_trace(value)
        else:
            redacted[key] = value
    return redacted


def snapshot_state(state: ConversationState) -> dict[str, Any]:
    return {
        "goal": state.conversation_goal,
        "product": state.product,
        "mode": state.mode,
        "lead_status": state.lead_status,
        "ticket_status": state.ticket_status,
    }


def format_chat_trace(state: ConversationState) -> str:
    trace = dict(state.trace or {})
    metadata = dict(state.retrieval_metadata or {})
    before = dict(trace.get("state_before") or {})
    preview = dict(trace.get("retrieval_preview") or {})
    user = _safe_text(state.user_message or "")
    original = _safe_text(str(trace.get("query_original") or state.user_message or ""))
    rewritten = _safe_text(state.query_rewritten or "")
    lines = [
        "====================================================",
        f"CHAT TRACE {state.conversation_id or '-'}",
        "====================================================",
        "",
        "USER",
        user,
        "",
        "STATE BEFORE",
        f"goal: {before.get('goal') or ConversationGoal.NONE}",
        f"product: {before.get('product') or ''}",
        "",
        "TURN",
        f"intent: {state.current_turn_intent or ''}",
        f"needs_rag: {str(bool(trace.get('should_retrieve'))).lower()}",
    ]
    if trace.get("should_retrieve") or state.query_rewritten or trace.get("retrieval_used"):
        lines.extend(
            [
                "",
                "QUERY",
                f"original: {original}",
                f"rewritten: {rewritten}",
            ]
        )
    if trace.get("retrieval_used") or preview:
        corpus = metadata.get("retrieval_version") or ""
        final_hits = list(preview.get("final") or [])
        k = metadata.get("final_context_count")
        if k is None:
            k = len(final_hits)
        lines.extend(["", "RETRIEVAL", f"corpus: {corpus}", f"k: {k}"])
        retrieval_hits = list(preview.get("vector") or final_hits)
        lines.extend(_format_hits(retrieval_hits))
        reranked = list(preview.get("reranked") or [])
        if reranked:
            lines.extend(["", "RERANK"])
            lines.extend(_format_hits(reranked))
        if final_hits:
            lines.extend(["", "FINAL CONTEXT"])
            for index, item in enumerate(final_hits, start=1):
                label = item.get("title") or item.get("section") or item.get("chunk_id") or ""
                section = item.get("section") or ""
                if section and section not in str(label):
                    label = f"{label} ({section})" if label else section
                lines.append(f"{index}. {label}")
    if trace.get("model_used") or trace.get("generation_temperature") is not None:
        lines.extend(
            [
                "",
                "GENERATION",
                f"model={trace.get('model_used') or ''}",
                f"temperature={trace.get('generation_temperature')}",
            ]
        )
    if "grounded" in trace:
        source_ids = trace.get("source_ids") or []
        lines.extend(
            [
                "",
                "GROUNDING",
                f"grounded={str(bool(trace.get('grounded'))).lower()}",
                f"source_ids={source_ids}",
            ]
        )
    if state.response:
        lines.extend(["", "FINAL", _safe_text(state.response)])
    lines.extend(
        [
            "",
            "STATE AFTER",
            f"goal={state.conversation_goal or ConversationGoal.NONE}",
            f"product={state.product or ''}",
        ]
    )
    return "\n".join(lines)


def log_trace(state: ConversationState) -> None:
    metadata = dict(state.retrieval_metadata or {})
    trace = dict(state.trace or {})
    safe = redact_trace(
        {
            "conversation_id": state.conversation_id,
            "mode": state.mode,
            "intent": state.intent,
            "conversation_goal": state.conversation_goal,
            "current_turn_intent": state.current_turn_intent,
            "lead_workflow": state.lead_workflow,
            "support_workflow": state.support_workflow,
            "workflow_state": (
                state.lead_workflow
                if state.conversation_goal == ConversationGoal.LEAD
                else state.support_workflow
                if state.conversation_goal == ConversationGoal.SUPPORT
                else ""
            ),
            "query_rewritten": bool(state.query_rewritten),
            "query_rewrite_used": bool(state.query_rewritten),
            "retrieval_used": bool(trace.get("retrieval_used")),
            "retrieval_started": bool(trace.get("retrieval_started")),
            "vector_candidate_count": metadata.get("vector_candidate_count"),
            "keyword_candidate_count": metadata.get("keyword_candidate_count"),
            "merged_candidate_count": metadata.get("merged_candidate_count"),
            "reranked_count": metadata.get("reranked_count"),
            "final_context_count": metadata.get("final_context_count"),
            "model_used": trace.get("model_used", ""),
            "generation_temperature": trace.get("generation_temperature"),
            "grounded": trace.get("grounded"),
            "tool_called": trace.get("tool_called", ""),
            "llm_call_count": trace.get("llm_call_count", 0),
            "query_rewrite_ms": trace.get("query_rewrite_ms"),
            "retrieval_ms": trace.get("retrieval_ms"),
            "generation_ms": trace.get("generation_ms"),
            "retrieval_stage_ms": trace.get("retrieval_stage_ms"),
            "total_latency_ms": trace.get("total_latency_ms"),
            "explicit_action": state.explicit_action,
            "capabilities": trace.get("capabilities") or [],
            "needs_natural_reply": bool(trace.get("needs_natural_reply")),
            "user_context_keys": sorted((state.user_context or {}).keys()),
            "lead_intent": bool(state.lead_intent),
            "support_intent": bool(state.support_intent),
        }
    )
    logger.info("conversation_trace %s", safe)


def _safe_text(text: str) -> str:
    return PHONE_RE.sub("[phone]", text or "")


def _format_hits(hits: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(hits, start=1):
        score = item.get("score")
        score_text = "" if score is None else f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
        lines.append("")
        lines.append(f"#{index}  score={score_text}")
        lines.append(f"chunk={item.get('chunk_id') or ''}")
        if item.get("section"):
            lines.append(f"section={item.get('section')}")
        text = (item.get("text") or "").replace("\n", " ")
        if text:
            lines.append(f"text={text}")
    return lines
