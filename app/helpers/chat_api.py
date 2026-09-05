from __future__ import annotations

from typing import Any

from app.services.conversation.models import ConversationState
from app.services.diagnostics.recorder import tracing_enabled


def chat_result_payload(result: ConversationState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "conversation_id": result.conversation_id,
        "mode": result.mode or "KNOWLEDGE",
        "response": result.response,
        "answer": result.response,
        "sources": [
            {
                "url": str(source.get("url") or ""),
                "title": str(source.get("title") or ""),
                "score": float(source.get("score") or 0.0),
            }
            for source in result.sources
        ],
    }
    trace_id = (result.trace or {}).get("trace_id")
    if trace_id and tracing_enabled():
        payload["debug_trace_id"] = trace_id
    return payload
