from __future__ import annotations

from app.services.conversation.models import ChatMode, ConversationGoal, LeadStatus, TicketStatus, TurnIntent
from app.services.conversation.router import ChatRouter
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


def test_1_knowledge_then_context_then_callback() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=knowledge)
    cid = "p21-1"
    first = orchestrator.handle(cid, "I want to buy TINY.")
    assert first.conversation_goal == ConversationGoal.LEAD
    assert first.lead_intent is True
    assert lead.leads == []
    assert "please provide your phone number" not in first.response.lower()
    orchestrator.handle(cid, "What is the warranty?")
    orchestrator.handle(cid, "What is the price?")
    info = orchestrator.handle(cid, "I'm Rahul from Delhi.")
    assert info.user_name == "Rahul"
    assert info.city == "Delhi"
    assert info.conversation_goal == ConversationGoal.LEAD
    assert lead.leads == []
    action = orchestrator.handle(cid, "Please call me.")
    assert action.conversation_goal == ConversationGoal.LEAD
    assert action.awaiting_field == "phone"
    assert lead.leads == []
    assert recorder.call_count >= 3
    assert any("warranty" in query.lower() for query in knowledge.queries)
    assert any("price" in query.lower() for query in knowledge.queries)


def test_2_mixed_context_product_correction() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=knowledge)
    cid = "p21-2"
    first = orchestrator.handle(cid, "I want to buy TINY. My father uses Signia.")
    assert first.conversation_goal == ConversationGoal.LEAD
    assert first.product == "TINY"
    assert first.user_context.get("relative_uses", "").lower().startswith("signia")
    diff = orchestrator.handle(cid, "Actually what is the difference between TINY and Radius M16?")
    assert recorder.call_count >= 2
    assert diff.conversation_goal == ConversationGoal.LEAD
    assert "difference" in knowledge.queries[-1].lower()
    updated = orchestrator.handle(cid, "I think Radius M16 is better.")
    assert updated.product == "Radius M16"
    assert updated.conversation_goal == ConversationGoal.LEAD
    assert updated.user_context.get("relative_uses", "").lower().startswith("signia")
    assert lead.leads == []


def test_3_support_then_rag_then_support() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, _, tickets, *_ = make_graph_stack(knowledge=knowledge)
    cid = "p21-3"
    first = orchestrator.handle(cid, "My Radius M16 isn't working.")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.product == "Radius M16"
    detail = orchestrator.handle(cid, "The sound is very low.")
    assert "low" in (detail.support_issue or "").lower()
    rag = orchestrator.handle(cid, "How does BTE work?")
    assert recorder.call_count >= 2
    assert knowledge.queries[-1] == "How does BTE work?"
    assert rag.conversation_goal == ConversationGoal.SUPPORT
    tried = orchestrator.handle(cid, "I tried the troubleshooting.")
    assert tried.conversation_goal == ConversationGoal.SUPPORT
    assert tried.mode == ChatMode.SUPPORT
    assert tickets.tickets == []


def test_4_voluntary_lead_fields_ask_only_missing() -> None:
    orchestrator, _, lead, *_ = make_graph_stack(knowledge=FakeKnowledge())
    result = orchestrator.handle("p21-4", "My name is Rahul, I'm from Delhi and I want TINY.")
    assert result.user_name == "Rahul"
    assert result.city == "Delhi"
    assert result.product == "TINY"
    assert result.awaiting_field == ""
    assert lead.leads == []
    assert "which city" not in result.response.lower()
    assert "phone" not in result.response.lower()


def test_5_why_buy_is_knowledge() -> None:
    from app.services.conversation.models import ConversationState

    routed = ChatRouter().route(ConversationState(user_message="Why should I buy from Earkart?"))
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.current_turn_intent == TurnIntent.KNOWLEDGE
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=FakeKnowledge())
    result = orchestrator.handle("p21-5", "Why should I buy from Earkart?")
    assert result.mode == ChatMode.KNOWLEDGE
    assert result.conversation_goal == ConversationGoal.KNOWLEDGE
    assert recorder.generation_call_count == 1
    assert lead.leads == []


def test_6_buy_sets_lead_objective_not_form() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=knowledge)
    result = orchestrator.handle("p21-6", "I want to buy TINY.")
    assert result.conversation_goal == ConversationGoal.LEAD
    assert result.lead_intent is True
    assert result.product == "TINY"
    assert result.awaiting_field == ""
    assert knowledge.queries == []
    assert lead.leads == []
    assert "please provide your phone number" not in result.response.lower()


def test_7_knowledge_then_buy_it() -> None:
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=FakeKnowledge())
    first = orchestrator.handle("p21-7", "What is TINY?")
    assert first.mode == ChatMode.KNOWLEDGE
    second = orchestrator.handle("p21-7", "I want to buy it.")
    assert second.conversation_goal == ConversationGoal.LEAD
    assert second.product == "TINY"
    assert lead.leads == []
    assert recorder.call_count >= 2


def test_8_user_context_is_not_company_knowledge() -> None:
    knowledge = FakeKnowledge()
    orchestrator, *_ = make_graph_stack(knowledge=knowledge)
    result = orchestrator.handle("p21-8", "I use Radius M16 and it usually lasts me all day.")
    assert result.user_context.get("owns") == "Radius M16"
    assert result.user_context.get("battery_experience") == "lasts all day"
    assert knowledge.queries == []
    assert not result.trace.get("retrieval_used")


def test_9_interrupt_answers_rag_first() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=knowledge)
    cid = "p21-9"
    first = orchestrator.handle(cid, "I want to buy TINY.")
    assert first.conversation_goal == ConversationGoal.LEAD
    mid = orchestrator.handle(cid, "Wait, before that, what is the warranty?")
    assert recorder.call_count >= 2
    assert "warranty" in knowledge.queries[-1].lower()
    assert mid.conversation_goal == ConversationGoal.LEAD
    assert mid.product == "TINY"
    assert mid.lead_status == LeadStatus.COLLECTING
    assert lead.leads == []
    assert "please provide your phone number first" not in mid.response.lower()


def test_10_mixed_lead_message_only_phone_missing() -> None:
    orchestrator, _, lead, *_ = make_graph_stack(knowledge=FakeKnowledge())
    result = orchestrator.handle(
        "p21-10",
        "I want to buy TINY and I'm Rahul from Delhi. Can someone call me?",
    )
    assert result.product == "TINY"
    assert result.user_name == "Rahul"
    assert result.city == "Delhi"
    assert result.lead_intent is True
    assert result.awaiting_field == "phone"
    assert not result.phone
    assert lead.leads == []
    assert result.lead_status != LeadStatus.CREATED


def test_tool_not_created_without_fields() -> None:
    orchestrator, _, lead, tickets, *_ = make_graph_stack(knowledge=FakeKnowledge())
    orchestrator.handle("p21-safe", "I want to buy TINY.")
    orchestrator.handle("p21-safe", "Please call me.")
    assert lead.leads == []
    orchestrator.handle("p21-sup", "My Radius M16 isn't working.")
    assert tickets.tickets == []
    assert orchestrator.handle("p21-sup", "Please create a support ticket.").ticket_status == TicketStatus.COLLECTING
