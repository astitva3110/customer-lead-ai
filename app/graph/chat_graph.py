from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.services.conversation.guardrail import GUARDRAIL_REJECTION_MESSAGE, guardrail_check
from app.services.conversation.knowledge_flow import KnowledgeFlow
from app.services.conversation.lead_service import LeadService
from app.services.conversation.models import ChatMode, ConversationState
from app.services.conversation.query_rewriter import apply_named_product
from app.services.conversation.router import ChatRouter
from app.services.conversation.support_service import SupportService
from app.services.diagnostics.recorder import current_trace, record_guardrail
from app.services.generation.generation_service import GenerationService


class GraphState(TypedDict, total=False):
    conversation_id: str
    user_message: str
    conversation_history: list[dict[str, str]]
    intent: str
    mode: str
    return_mode: str
    awaiting_field: str
    conversation_goal: str
    current_turn_intent: str
    lead_workflow: str
    support_workflow: str
    lead_intent: bool
    support_intent: bool
    sales_interest: bool
    lead_stage: str
    lead_collection_active: bool
    support_collection_active: bool
    explicit_action: str
    user_context: dict[str, Any]
    product: str
    product_id: str
    user_name: str
    phone: str
    country: str
    phone_country: str
    city: str
    city_country: str
    session_country: str
    pending_phone: str
    support_issue: str
    retrieved_context: list[dict[str, Any]]
    retrieved_chunk_ids: list[str]
    retrieval_metadata: dict[str, Any]
    lead_status: str
    ticket_status: str
    response: str
    sources: list[dict[str, Any]]
    query_rewritten: str
    guardrail_rejected: bool
    error: str
    trace: dict[str, Any]


def _load(payload: GraphState) -> ConversationState:
    return ConversationState.from_dict(dict(payload))


def _dump(state: ConversationState) -> GraphState:
    return state.to_dict()


def build_chat_graph(
    *,
    router: ChatRouter,
    knowledge: KnowledgeFlow,
    lead: LeadService,
    support: SupportService,
    generation: GenerationService,
):
    """LangGraph owns conversation workflow. Retrieval stays outside the graph."""

    def guardrail_node(payload: GraphState) -> GraphState:
        state = _load(payload)
        started = time.perf_counter()
        reason = guardrail_check(state.user_message)
        if current_trace():
            record_guardrail(
                executed=True,
                result=reason,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        history = list(state.conversation_history)
        history.append({"role": "user", "content": state.user_message})
        state.conversation_history = history
        if reason:
            state.guardrail_rejected = True
            state.mode = ChatMode.REJECTED
            state.intent = "rejected"
            state.response = GUARDRAIL_REJECTION_MESSAGE
            state.trace["guardrail"] = reason
            return _dump(state)
        apply_named_product(state)
        return _dump(state)

    def router_node(payload: GraphState) -> GraphState:
        started = time.perf_counter()
        state = router.route(_load(payload))
        if current_trace():
            state.trace = dict(state.trace or {})
            state.trace["routing_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return _dump(state)

    def knowledge_node(payload: GraphState) -> GraphState:
        return _dump(knowledge.handle(_load(payload)))

    def lead_node(payload: GraphState) -> GraphState:
        return _dump(lead.converse(_load(payload)))

    def support_node(payload: GraphState) -> GraphState:
        return _dump(support.converse(_load(payload)))

    def restore_mode_node(payload: GraphState) -> GraphState:
        state = _load(payload)
        if state.return_mode:
            state.mode = state.return_mode
            state.return_mode = ""
        return _dump(state)

    def reply_node(payload: GraphState) -> GraphState:
        return _dump(generation.converse(_load(payload)))

    def finish_node(payload: GraphState) -> GraphState:
        state = _load(payload)
        if state.response:
            history = list(state.conversation_history)
            history.append({"role": "assistant", "content": state.response})
            state.conversation_history = history
        return _dump(state)

    def after_guardrail(payload: GraphState) -> str:
        return "finish" if payload.get("guardrail_rejected") else "router"

    def after_router(payload: GraphState) -> str:
        trace = payload.get("trace") or {}
        if trace.get("should_retrieve"):
            return "knowledge"
        if trace.get("needs_natural_reply"):
            return "reply"
        mode = payload.get("mode")
        if mode == ChatMode.LEAD:
            return "lead"
        if mode == ChatMode.SUPPORT:
            return "support"
        return "reply"

    def after_knowledge(payload: GraphState) -> str:
        trace = payload.get("trace") or {}
        if trace.get("needs_natural_reply") and not str(payload.get("response") or "").strip():
            return "reply"
        return "restore_mode" if payload.get("return_mode") else "finish"

    def after_business(payload: GraphState) -> str:
        trace = payload.get("trace") or {}
        if trace.get("needs_natural_reply") and not str(payload.get("response") or "").strip():
            return "reply"
        return "finish"

    builder = StateGraph(GraphState)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("router", router_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("lead", lead_node)
    builder.add_node("support", support_node)
    builder.add_node("restore_mode", restore_mode_node)
    builder.add_node("reply", reply_node)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        after_guardrail,
        {"finish": "finish", "router": "router"},
    )
    builder.add_conditional_edges(
        "router",
        after_router,
        {"knowledge": "knowledge", "lead": "lead", "support": "support", "reply": "reply"},
    )
    builder.add_conditional_edges(
        "knowledge",
        after_knowledge,
        {"restore_mode": "restore_mode", "finish": "finish", "reply": "reply"},
    )
    builder.add_edge("restore_mode", "finish")
    builder.add_conditional_edges(
        "lead",
        after_business,
        {"reply": "reply", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "support",
        after_business,
        {"reply": "reply", "finish": "finish"},
    )
    builder.add_edge("reply", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
