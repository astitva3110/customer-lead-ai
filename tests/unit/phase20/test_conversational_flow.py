from __future__ import annotations

from app.services.conversation.models import ChatMode, ConversationGoal, LeadStatus, TicketStatus, TurnIntent
from app.services.conversation.router import ChatRouter
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


def test_a_lead_plus_knowledge_keeps_goal() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=knowledge)
    first = orchestrator.handle("p20-a", "I want to buy TINY.")
    assert first.conversation_goal == ConversationGoal.LEAD
    assert first.current_turn_intent == TurnIntent.LEAD_INTENT
    assert first.mode == ChatMode.LEAD
    assert lead.leads == []
    assert "please provide your phone number" not in first.response.lower()
    rag = orchestrator.handle("p20-a", "What is TINY?")
    assert knowledge.queries[-1] == "What is TINY?"
    assert recorder.call_count >= 2
    assert rag.conversation_goal == ConversationGoal.LEAD
    assert rag.current_turn_intent == TurnIntent.KNOWLEDGE
    assert rag.mode == ChatMode.LEAD
    assert rag.product == "TINY"
    assert rag.lead_status == LeadStatus.COLLECTING


def test_b_lead_plus_warranty_keeps_goal() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, *_ = make_graph_stack(knowledge=knowledge)
    orchestrator.handle("p20-b", "I want to buy TINY.")
    result = orchestrator.handle("p20-b", "What is the warranty?")
    assert recorder.call_count >= 2
    assert "warranty" in knowledge.queries[-1].lower() or knowledge.queries[-1]
    assert result.conversation_goal == ConversationGoal.LEAD
    assert result.mode == ChatMode.LEAD
    assert result.product == "TINY"


def test_c_lead_product_change() -> None:
    orchestrator, *_ = make_graph_stack(knowledge=FakeKnowledge())
    orchestrator.handle("p20-c", "I want to buy TINY.")
    result = orchestrator.handle("p20-c", "Actually I want Radius M16.")
    assert result.product == "Radius M16"
    assert result.conversation_goal == ConversationGoal.LEAD


def test_d_lead_information_extraction() -> None:
    orchestrator, _, lead, *_ = make_graph_stack(knowledge=FakeKnowledge())
    result = orchestrator.handle("p20-d", "I am Rahul from Delhi and I want to buy TINY.")
    assert result.product == "TINY"
    assert result.user_name == "Rahul"
    assert result.city == "Delhi"
    assert result.awaiting_field == ""
    assert not result.phone
    assert lead.leads == []
    assert "phone" not in result.response.lower()


def test_e_lead_interrupted_while_collecting_phone() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=knowledge)
    first = orchestrator.handle("p20-e", "I want to buy TINY.")
    assert first.conversation_goal == ConversationGoal.LEAD
    mid = orchestrator.handle("p20-e", "What is the delivery time?")
    assert recorder.call_count >= 2
    assert mid.conversation_goal == ConversationGoal.LEAD
    assert mid.lead_status == LeadStatus.COLLECTING
    assert mid.product == "TINY"
    assert lead.leads == []
    assert "please provide your phone number first" not in mid.response.lower()


def test_f_support_plus_knowledge_keeps_goal() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, *_ = make_graph_stack(knowledge=knowledge)
    first = orchestrator.handle("p20-f", "My Radius M16 is not working.")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.product == "Radius M16"
    assert first.support_issue
    mid = orchestrator.handle("p20-f", "How does BTE work?")
    assert recorder.call_count >= 2
    assert knowledge.queries[-1] == "How does BTE work?"
    assert mid.conversation_goal == ConversationGoal.SUPPORT
    assert mid.mode == ChatMode.SUPPORT
    assert mid.support_issue == first.support_issue
    assert mid.product == "Radius M16"


def test_g_support_information_extraction() -> None:
    orchestrator, _, _, tickets, *_ = make_graph_stack(knowledge=FakeKnowledge())
    result = orchestrator.handle(
        "p20-g",
        "My name is Rahul, I use Radius M16 and it has very low sound.",
    )
    assert result.user_name == "Rahul"
    assert result.product == "Radius M16"
    assert "low sound" in result.support_issue.lower()
    assert tickets.tickets == []
    assert "what name should we put" not in result.response.lower()


def test_h_support_interruption_returns_to_support() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, *_ = make_graph_stack(knowledge=knowledge)
    orchestrator.handle("p20-h", "My hearing aid isn't working.")
    result = orchestrator.handle("p20-h", "What is TINY?")
    assert recorder.call_count >= 2
    assert result.conversation_goal == ConversationGoal.SUPPORT
    assert result.mode == ChatMode.SUPPORT
    assert result.ticket_status == TicketStatus.COLLECTING


def test_i_knowledge_to_lead() -> None:
    orchestrator, recorder, lead, *_ = make_graph_stack(knowledge=FakeKnowledge())
    first = orchestrator.handle("p20-i", "What is TINY?")
    assert first.mode == ChatMode.KNOWLEDGE
    assert recorder.call_count == 1
    second = orchestrator.handle("p20-i", "I want to buy it.")
    assert second.mode == ChatMode.LEAD
    assert second.conversation_goal == ConversationGoal.LEAD
    assert second.product == "TINY"
    assert lead.leads == []


def test_j_knowledge_to_support() -> None:
    orchestrator, *_ = make_graph_stack(knowledge=FakeKnowledge())
    orchestrator.handle("p20-j", "What is Radius M16?")
    result = orchestrator.handle("p20-j", "My Radius M16 isn't working.")
    assert result.mode == ChatMode.SUPPORT
    assert result.conversation_goal == ConversationGoal.SUPPORT
    assert result.product == "Radius M16"


def test_k_purchase_language_split() -> None:
    from app.services.conversation.models import ConversationState

    why = ChatRouter().route(ConversationState(user_message="Why should I buy from Earkart?"))
    assert why.mode == ChatMode.KNOWLEDGE
    assert why.current_turn_intent == TurnIntent.KNOWLEDGE
    buy = ChatRouter().route(ConversationState(user_message="I want to buy from Earkart."))
    assert buy.mode == ChatMode.LEAD
    assert buy.current_turn_intent == TurnIntent.LEAD_INTENT
    how = ChatRouter().route(ConversationState(user_message="How can I buy TINY?"))
    assert how.mode == ChatMode.LEAD
    benefits = ChatRouter().route(ConversationState(user_message="What are the benefits of buying TINY?"))
    assert benefits.mode == ChatMode.KNOWLEDGE


def test_l_corpus_gap_no_hallucination() -> None:
    knowledge = FakeKnowledge()
    orchestrator, *_ = make_graph_stack(knowledge=knowledge)
    result = orchestrator.handle("p20-l", "What is the battery life in hours?")
    assert result.response == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.sources == []
