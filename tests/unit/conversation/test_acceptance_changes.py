from __future__ import annotations

from app.helpers.conversation_reply import lead_created_reply
from app.helpers.state_manager import conversation_status_label, current_status_label
from app.services.conversation.models import ChatMode, ConversationGoal, ConversationState, LeadStatus
from app.services.conversation.query_rewriter import QueryRewriter, normalize_for_routing
from app.services.conversation.router import ChatRouter
from tests.unit.conversation.fakes import FakeKnowledge, make_orchestrator


def _complete_lead(orchestrator, cid: str, name: str = "Titititi") -> None:
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Please call me.")
    orchestrator.handle(cid, name)
    orchestrator.handle(cid, "+91 9876543210")
    done = orchestrator.handle(cid, "India")
    assert done.lead_status == LeadStatus.CREATED
    assert done.response == lead_created_reply(name)


def test_01_purchase_intent_routes_to_lead() -> None:
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("acc-1", "I want to buy TINY")
    assert conversation_status_label(result) == "SALE"
    assert current_status_label(result) == "LEAD"
    assert result.mode == ChatMode.LEAD


def test_02_rewriter_normalizes_tinny_to_tiny() -> None:
    rewritten = normalize_for_routing("i need to buy tinny")
    assert rewritten.rewritten_query == "I need to buy TINY"
    assert "TINY" in rewritten.entities
    router = ChatRouter()
    state = ConversationState(user_message="i need to buy tinny")
    QueryRewriter().normalize(state)
    routed = router.route(state)
    assert routed.mode == ChatMode.LEAD


def test_03_mixed_purchase_and_warranty_routes_to_knowledge() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("acc-3", "I want to buy TINY. What is its warranty?")
    assert conversation_status_label(result) == "SALE"
    assert result.trace.get("current_status") == "KNOWLEDGE"
    assert knowledge.queries
    assert any("warranty" in query.lower() for query in knowledge.queries)
    assert result.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES}


def test_04_after_lead_completion_general_does_not_repeat_completion() -> None:
    orchestrator, *_ = make_orchestrator()
    cid = "acc-4"
    _complete_lead(orchestrator, cid)
    result = orchestrator.handle(cid, "how are u")
    assert current_status_label(result) == "GENERAL"
    assert result.response != lead_created_reply("Titititi")
    assert "shared with our sales team" not in (result.response or "").lower()


def test_05_after_lead_completion_knowledge_routes_to_rag() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "acc-5"
    _complete_lead(orchestrator, cid)
    result = orchestrator.handle(cid, "I want to know about Bluup.")
    assert current_status_label(result) == "KNOWLEDGE"
    assert knowledge.queries
    assert any("bluup" in query.lower() for query in knowledge.queries)
    assert "shared with our sales team" not in (result.response or "").lower()


def test_06_rewriter_normalizes_warranty_spelling() -> None:
    state = ConversationState(user_message="what is tinny warrenty?", product="TINY")
    QueryRewriter().normalize(state)
    assert state.query_rewritten == "What is TINY's warranty?"
    router = ChatRouter()
    routed = router.route(state)
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True


def test_07_chat_trace_still_records_turn(monkeypatch, tmp_path) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "chat_trace_enabled", True)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("acc-7", "What is TINY?")
    assert result.trace.get("trace_id")
    assert result.response


def test_08_observability_metadata_shape() -> None:
    from app.services.diagnostics.langsmith_tracing import add_token_usage
    from app.services.diagnostics.models import ChatTrace

    trace = ChatTrace()
    trace.request = {"trace_id": "trace-obs"}
    add_token_usage(trace, {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20})
    assert trace.observability["token_usage"]["input_tokens"] == 12
    assert trace.observability["token_usage"]["output_tokens"] == 8


def test_09_factual_questions_use_rag_not_general() -> None:
    from app.helpers.state_manager import current_status_label

    orchestrator, knowledge, llm, *_ = make_orchestrator()
    cid = "acc-9"
    orchestrator.handle(cid, "Hi")
    assert knowledge.queries == []
    result = orchestrator.handle(cid, "Who is the CEO of Earkart?")
    assert current_status_label(result) == "KNOWLEDGE"
    assert knowledge.queries
    assert any("ceo" in query.lower() or "who" in query.lower() for query in knowledge.queries)
    assert result.trace.get("should_retrieve") is True
    assert llm.last_temperature == 0.0


def test_10_missing_rag_evidence_does_not_use_general_knowledge() -> None:
    from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE

    orchestrator, knowledge, llm, *_ = make_orchestrator(knowledge=FakeKnowledge(chunks=[]))
    result = orchestrator.handle("acc-10", "Who founded Earkart?")
    assert knowledge.queries
    assert result.response == INSUFFICIENT_INFORMATION_MESSAGE
    assert llm.generation_call_count == 0
    assert result.sources == []
