"""Entity → HTTP payload dicts. No Pydantic, no I/O."""

from __future__ import annotations

from typing import Any

from app.domain.entities import ChatConversation, ChatConversationBrief, ChatTurn, RetrievalLayerHit
from app.helpers.chat_trace_persist import group_hits_by_layer, isoformat_utc


def conversation_brief_payload(item: ChatConversationBrief) -> dict[str, Any]:
    return {
        "conversation_id": item.conversation_id,
        "last_trace_id": item.last_trace_id,
        "last_user_message": item.last_user_message,
        "last_response": item.last_response,
        "last_intent": item.last_intent,
        "last_total_ms": item.last_total_ms,
        "turn_count": item.turn_count,
        "updated_at": isoformat_utc(item.updated_at),
    }


def conversation_detail_payload(item: ChatConversation) -> dict[str, Any]:
    return {
        "conversation_id": item.conversation_id,
        "turns": [chat_turn_detail_payload(turn) for turn in item.turns],
    }


def chat_turn_detail_payload(turn: ChatTurn) -> dict[str, Any]:
    latency = turn.latency
    return {
        "trace_id": turn.trace_id,
        "conversation_id": turn.conversation_id,
        "user_message": turn.user_message,
        "rewritten_query": turn.rewritten_query,
        "intent": turn.intent,
        "response": turn.response,
        "tool_name": turn.tool_name,
        "tool_success": turn.tool_success,
        "lead_id": turn.lead_id,
        "ticket_id": turn.ticket_id,
        "backend": turn.backend,
        "reranker_name": turn.reranker_name,
        "latency": {
            "guardrail_ms": latency.guardrail_ms,
            "routing_ms": latency.routing_ms,
            "rewrite_ms": latency.rewrite_ms,
            "retrieval_ms": latency.retrieval_ms,
            "generation_ms": latency.generation_ms,
            "tool_ms": latency.tool_ms,
            "total_ms": latency.total_ms,
            "vector_ms": latency.vector_ms,
            "keyword_ms": latency.keyword_ms,
            "merge_ms": latency.merge_ms,
            "rerank_ms": latency.rerank_ms,
            "threshold_ms": latency.threshold_ms,
        },
        "layers": {
            layer: [_hit_payload(hit) for hit in rows]
            for layer, rows in group_hits_by_layer(turn.hits).items()
        },
        "created_at": isoformat_utc(turn.created_at),
    }


def _hit_payload(hit: RetrievalLayerHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "title": hit.title,
        "section_path": hit.section_path,
        "vector_score": hit.vector_score,
        "keyword_score": hit.keyword_score,
        "rerank_score": hit.rerank_score,
        "combined_score": hit.combined_score,
        "original_retrieval_rank": hit.original_retrieval_rank,
        "text_preview": hit.text_preview,
    }
