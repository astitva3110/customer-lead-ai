from __future__ import annotations

from app.evaluation.client import TurnOutcome
from app.evaluation.schema import ConversationExpected, GeneratedConversation, TurnMessage, conversation_id_for
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE


def make_conversation(**overrides) -> GeneratedConversation:
    messages = overrides.pop(
        "messages",
        [
            TurnMessage(turn=1, text="What is TINY?", needs_rag=True, answerable=True, knowledge_keys=["product:tiny"]),
            TurnMessage(
                turn=2,
                text="What is its warranty?",
                needs_rag=True,
                needs_rewrite=True,
                answerable=True,
                knowledge_keys=["policy:warranty"],
                expected_product="TINY",
            ),
            TurnMessage(turn=3, text="Okay.", expected_intent="CONFIRMATION"),
        ],
    )
    expected = overrides.pop(
        "expected",
        ConversationExpected(
            answerable=True,
            knowledge_keys=["product:tiny", "policy:warranty"],
            goal="KNOWLEDGE",
            rag_turns=[1, 2],
            rewrite_turns=[2],
            expected_product="TINY",
        ),
    )
    payload = {
        "conversation_id": overrides.pop("conversation_id", conversation_id_for(1)),
        "category": overrides.pop("category", "KNOWLEDGE_FOLLOWUP"),
        "language": "en",
        "messages": messages,
        "expected": expected,
    }
    payload.update(overrides)
    return GeneratedConversation.model_validate(payload)


def knowledge_trace(
    *,
    blocked=False,
    intent="KNOWLEDGE",
    needs_rag=True,
    rewrite=False,
    product="TINY",
    raw=None,
    reranked=None,
    final=None,
    grounded=True,
    reason="ok",
    response="TINY is compact.",
    tool=None,
    goal="KNOWLEDGE",
):
    raw = raw if raw is not None else [
        {
            "rank": 1,
            "chunk_id": "c-tiny",
            "title": "TINY",
            "section_path": "6.1 TINY",
            "text_preview": "TINY features: rechargeable 16-channel",
        }
    ]
    final = final if final is not None else [
        {
            "position": 1,
            "chunk_id": "c-tiny",
            "title": "TINY",
            "section_path": ["6.1 TINY"],
            "text": "TINY features: rechargeable 16-channel",
        }
    ]
    rerank = {"enabled": True, "candidates": reranked if reranked is not None else raw}
    return {
        "trace_id": "tr-test",
        "guardrail": {"executed": True, "blocked": blocked, "reason": "prompt_injection" if blocked else None},
        "turn_understanding": {
            "turn_intent": intent,
            "needs_rag": needs_rag,
            "information_updates": {},
            "user_context_updates": {},
            "method": "deterministic",
        },
        "query": {"rewrite_executed": rewrite, "rewritten_query": "warranty of TINY" if rewrite else None},
        "retrieval": {"candidates": raw, "raw_count": len(raw)},
        "reranking": rerank,
        "final_context": {"chunks": final, "context_count": len(final or [])},
        "generation": {"llm_call_count": 1, "model": "test", "validator_reason": reason},
        "grounding": {
            "grounded_returned": grounded,
            "validator_result": grounded and reason == "ok",
            "validator_reason": reason,
            "fallback_triggered": not grounded,
            "invalid_source_ids": [],
        },
        "response": {"final_response": response, "final_mode": "REJECTED" if blocked else "KNOWLEDGE"},
        "state_before": {"conversation_goal": goal, "current_product": product},
        "state_after": {
            "conversation_goal": goal,
            "current_turn_intent": intent,
            "current_product": product,
            "lead_status": "IDLE",
            "support_status": "IDLE",
            "user_context_keys": [],
            "user_context": {},
        },
        "tool_execution": tool or {"tool_executed": False},
        "latency": {"total_ms": 120.0},
    }


def passing_handle(conversation_id: str, message: str) -> TurnOutcome:
    rewrite = "its" in message.lower() or "warranty" in message.lower()
    trace = knowledge_trace(rewrite=rewrite)
    lowered = message.lower()
    if "ignore previous" in lowered or "ignore all previous" in lowered or "kill yourself" in lowered:
        trace = knowledge_trace(blocked=True, response="naa munna naaa")
    if "signia" in lowered or "bluetooth" in lowered or "aptx" in lowered:
        trace = knowledge_trace(
            raw=[],
            final=[],
            grounded=False,
            reason="llm_ungrounded",
            response=INSUFFICIENT_INFORMATION_MESSAGE,
        )
    return TurnOutcome(
        conversation_id=conversation_id,
        response=trace["response"]["final_response"],
        trace_id="tr-test",
        trace=trace,
    )
