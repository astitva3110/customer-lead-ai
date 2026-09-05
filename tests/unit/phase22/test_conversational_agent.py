from __future__ import annotations

from app.services.conversation.models import (
    ChatMode,
    ConversationGoal,
    ConversationState,
    LeadStatus,
    TurnIntent,
)
from app.services.conversation.router import ChatRouter
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.helpers.conversation_reply import lead_created_reply, ticket_created_reply
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


def _stack():
    knowledge = FakeKnowledge()
    orchestrator, recorder, lead, tickets, store, provider = make_graph_stack(knowledge=knowledge)
    return orchestrator, recorder, lead, tickets, store, provider, knowledge


def test_01_greeting_then_knowledge() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    cid = "p22-1"
    hi = orchestrator.handle(cid, "Hi")
    assert knowledge.queries == []
    assert "hey" in hi.response.lower() or "help" in hi.response.lower()
    assert "earkart is" not in hi.response.lower()
    asked = orchestrator.handle(cid, "What is Earkart?")
    assert knowledge.queries[-1] == "What is Earkart?"
    assert asked.mode == ChatMode.KNOWLEDGE
    assert asked.response != hi.response
    assert recorder.calls[-1]["temperature"] == 0.0


def test_02_knowledge_then_sales() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-2"
    first = orchestrator.handle(cid, "What is TINY?")
    assert first.mode == ChatMode.KNOWLEDGE
    second = orchestrator.handle(cid, "I want to buy it.")
    assert second.conversation_goal == ConversationGoal.LEAD
    assert second.product == "TINY"
    assert second.awaiting_field == ""
    assert lead.leads == []
    assert knowledge.queries == ["What is TINY?"]


def test_03_sales_knowledge_sales() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-3"
    buy = orchestrator.handle(cid, "I want to buy TINY.")
    assert buy.conversation_goal == ConversationGoal.LEAD
    rag = orchestrator.handle(cid, "How much is it?")
    assert knowledge.queries
    assert "how much" in knowledge.queries[-1].lower() or "TINY" in knowledge.queries[-1]
    assert rag.conversation_goal == ConversationGoal.LEAD
    assert "please provide your phone number" not in rag.response.lower()
    again = orchestrator.handle(cid, "Okay, can someone call me?")
    assert again.conversation_goal == ConversationGoal.LEAD
    assert again.awaiting_field == "name"
    assert lead.leads == []


def test_04_sales_knowledge_then_lead_creation() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-4"
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Does it have warranty?")
    assert any("warranty" in query.lower() for query in knowledge.queries)
    orchestrator.handle(cid, "Can someone call me?")
    orchestrator.handle(cid, "+91 9876543210")
    orchestrator.handle(cid, "Rahul")
    result = orchestrator.handle(cid, "Delhi")
    assert result.response == lead_created_reply("Rahul")
    assert result.lead_status == LeadStatus.CREATED
    assert lead.leads[0].product == "TINY"
    assert lead.leads[0].city == "Delhi"


def test_05_support_knowledge_support() -> None:
    orchestrator, _, _, tickets, *_rest, knowledge = _stack()
    cid = "p22-5"
    first = orchestrator.handle(cid, "My hearing aid isn't working.")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    rag = orchestrator.handle(cid, "How does BTE work?")
    assert knowledge.queries[-1] == "How does BTE work?"
    assert rag.conversation_goal == ConversationGoal.SUPPORT
    back = orchestrator.handle(cid, "The sound is very low.")
    assert back.conversation_goal == ConversationGoal.SUPPORT
    assert "low" in (back.support_issue or "").lower()
    assert tickets.tickets == []
    assert "phone number" not in rag.response.lower()


def test_06_support_troubleshooting_then_ticket() -> None:
    orchestrator, _, _, tickets, *_rest, knowledge = _stack()
    cid = "p22-6"
    orchestrator.handle(cid, "My Radius M16 isn't working.")
    tried = orchestrator.handle(cid, "What can I try?")
    assert knowledge.queries
    assert tried.conversation_goal == ConversationGoal.SUPPORT
    orchestrator.handle(cid, "I already cleaned it.")
    orchestrator.handle(cid, "Can you create a support ticket?")
    orchestrator.handle(cid, "Rahul")
    result = orchestrator.handle(cid, "+91 9876543210")
    assert result.response == ticket_created_reply("Rahul")
    assert tickets.tickets
    assert tickets.tickets[0].product == "Radius M16"


def test_07_user_volunteers_information() -> None:
    orchestrator, _, lead, *_ = _stack()
    cid = "p22-7"
    orchestrator.handle(cid, "I want to buy TINY.")
    info = orchestrator.handle(cid, "I'm Rahul from Delhi.")
    assert info.user_name == "Rahul"
    assert info.city == "Delhi"
    assert info.awaiting_field == ""
    assert lead.leads == []
    assert "which city" not in info.response.lower()
    assert "what name" not in info.response.lower()


def test_08_user_changes_product() -> None:
    orchestrator, *_ = _stack()
    cid = "p22-8"
    orchestrator.handle(cid, "I'm interested in TINY.")
    later = orchestrator.handle(cid, "Actually, I mean Radius M16.")
    assert later.product == "Radius M16"
    assert later.conversation_goal == ConversationGoal.LEAD
    pronoun = orchestrator.handle(cid, "What about its battery?")
    assert pronoun.product == "Radius M16"


def test_09_short_yes_uses_previous_question() -> None:
    orchestrator, recorder, lead, *_rest, knowledge = _stack()
    cid = "p22-9"
    first = orchestrator.handle(cid, "I want to buy TINY.")
    yes = orchestrator.handle(cid, "yes")
    assert yes.response != first.response
    assert "would you like to know anything about tiny before" not in yes.response.lower()
    assert yes.conversation_goal == ConversationGoal.LEAD
    assert yes.current_turn_intent == TurnIntent.CONFIRMATION
    assert yes.trace.get("should_retrieve") is not True
    assert not knowledge.queries
    assert lead.leads == []
    assert recorder.calls[-1]["temperature"] == 0.4


def test_10_short_no_uses_previous_question() -> None:
    orchestrator, _, lead, *_ = _stack()
    cid = "p22-10"
    orchestrator.handle(cid, "I want to buy TINY.")
    asked = orchestrator.handle(cid, "Can someone call me?")
    assert asked.awaiting_field == "name"
    refused = orchestrator.handle(cid, "no")
    assert refused.awaiting_field == ""
    assert refused.response != asked.response
    assert lead.leads == []
    assert "phone number should our team" not in refused.response.lower()


def test_11_mixed_sales_and_knowledge() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    result = orchestrator.handle(
        "p22-11",
        "I want to buy TINY, but first tell me how much it costs and whether it has warranty.",
    )
    assert result.conversation_goal == ConversationGoal.LEAD
    assert result.product == "TINY"
    assert knowledge.queries
    assert result.awaiting_field == ""
    assert lead.leads == []
    assert "phone" not in result.response.lower()
    assert "RAG" in (result.trace.get("capabilities") or [])


def test_12_mixed_support_and_knowledge() -> None:
    orchestrator, _, _, tickets, *_rest, knowledge = _stack()
    result = orchestrator.handle(
        "p22-12",
        "My hearing aid isn't working, I use Radius M16, and what can I try?",
    )
    assert result.conversation_goal == ConversationGoal.SUPPORT
    assert result.product == "Radius M16"
    assert knowledge.queries
    assert tickets.tickets == []
    assert "what name should we put" not in result.response.lower()


def test_13_user_changes_subject() -> None:
    orchestrator, *_rest, knowledge = _stack()
    cid = "p22-13"
    orchestrator.handle(cid, "I want to buy TINY.")
    changed = orchestrator.handle(cid, "What is Earkart?")
    assert changed.conversation_goal == ConversationGoal.LEAD
    assert knowledge.queries[-1] == "What is Earkart?"
    assert "please provide your phone number" not in changed.response.lower()


def test_14_company_question_during_sales() -> None:
    orchestrator, *_rest, knowledge = _stack()
    cid = "p22-14"
    orchestrator.handle(cid, "I want to buy TINY.")
    asked = orchestrator.handle(cid, "Why should I buy from Earkart?")
    assert asked.conversation_goal == ConversationGoal.LEAD
    assert asked.mode == ChatMode.LEAD
    assert knowledge.queries[-1] == "Why should I buy from Earkart?"


def test_15_product_question_during_support() -> None:
    orchestrator, *_rest, knowledge = _stack()
    cid = "p22-15"
    orchestrator.handle(cid, "My Radius M16 isn't working.")
    asked = orchestrator.handle(cid, "What does this button do?")
    assert asked.conversation_goal == ConversationGoal.SUPPORT
    assert knowledge.queries
    assert "phone number should support" not in asked.response.lower()


def test_16_several_questions_before_callback() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-16"
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "How much is it?")
    orchestrator.handle(cid, "Does it have warranty?")
    cont = orchestrator.handle(cid, "Okay sounds good.")
    assert cont.conversation_goal == ConversationGoal.LEAD
    assert lead.leads == []
    assert cont.awaiting_field == ""
    action = orchestrator.handle(cid, "Can someone call me?")
    assert action.awaiting_field == "name"
    assert len(knowledge.queries) >= 2


def test_17_name_city_phone_in_one_message() -> None:
    orchestrator, _, lead, *_ = _stack()
    result = orchestrator.handle(
        "p22-17",
        "I want to buy TINY. My name is Rahul, I'm from Delhi, call me on +91 9876543210.",
    )
    assert result.user_name == "Rahul"
    assert result.city == "Delhi"
    assert result.phone
    assert result.lead_status == LeadStatus.CREATED
    assert result.response == lead_created_reply("Rahul")
    assert len(lead.leads) == 1


def test_18_user_refuses_to_provide_phone() -> None:
    orchestrator, _, lead, *_ = _stack()
    cid = "p22-18"
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Please call me.")
    refused = orchestrator.handle(cid, "I don't want to share my number")
    assert refused.awaiting_field == ""
    assert lead.leads == []
    assert "what phone number" not in refused.response.lower()
    assert refused.conversation_goal == ConversationGoal.LEAD


def test_19_user_returns_to_previous_topic() -> None:
    orchestrator, *_rest, knowledge = _stack()
    cid = "p22-19"
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Does it have warranty?")
    orchestrator.handle(cid, "What is Earkart?")
    back = orchestrator.handle(cid, "What about its battery?")
    assert back.product == "TINY"
    assert back.conversation_goal == ConversationGoal.LEAD
    assert "battery" in knowledge.queries[-1].lower() or "TINY" in knowledge.queries[-1]


def test_20_greeting_in_middle_of_conversation() -> None:
    orchestrator, *_rest, knowledge = _stack()
    cid = "p22-20"
    first = orchestrator.handle(cid, "I want to buy TINY.")
    mid = orchestrator.handle(cid, "hi")
    assert mid.conversation_goal == ConversationGoal.LEAD
    assert mid.product == "TINY"
    assert knowledge.queries == []
    assert mid.response != first.response
    assert "please provide your phone number" not in mid.response.lower()


def test_21_tool_failure_does_not_claim_success() -> None:
    class _Broken:
        def create_lead(self, lead):
            del lead
            raise RuntimeError("crm down")

    knowledge = FakeKnowledge()
    orchestrator, _, *_ = make_graph_stack(knowledge=knowledge, lead_tool=_Broken())
    cid = "p22-21"
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Please call me.")
    orchestrator.handle(cid, "+91 9876543210")
    orchestrator.handle(cid, "Rahul")
    failed = orchestrator.handle(cid, "Delhi")
    assert failed.lead_status == LeadStatus.FAILED
    assert "details have been shared" not in failed.response.lower()
    assert "could not create" in failed.response.lower()


def test_22_no_rag_conversational_turns() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    cid = "p22-22"
    orchestrator.handle(cid, "Hi")
    thanks = orchestrator.handle(cid, "thanks")
    assert knowledge.queries == []
    assert thanks.response
    from tests.validation.harness import _is_semantic_router_call

    temps = [call["temperature"] for call in recorder.calls if not _is_semantic_router_call(call)]
    assert temps
    assert all(temp == 0.4 for temp in temps)


def test_23_rag_grounding_regression() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    result = orchestrator.handle("p22-23", "What is TINY?")
    assert result.sources
    assert result.response != INSUFFICIENT_INFORMATION_MESSAGE
    assert recorder.calls[-1]["temperature"] == 0.0
    assert knowledge.queries == ["What is TINY?"]


def test_24_no_hallucination_when_kb_lacks_information() -> None:
    orchestrator, *_ = _stack()
    result = orchestrator.handle("p22-24", "What is the Signia hearing aid?")
    assert result.response == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.sources == []


def test_router_greeting_and_how_much() -> None:
    router = ChatRouter()
    hi = router.route(ConversationState(user_message="hello"))
    assert hi.trace.get("should_retrieve") is False
    assert hi.trace.get("needs_natural_reply") is True
    sales = ConversationState(
        user_message="How much is it?",
        mode=ChatMode.LEAD,
        conversation_goal=ConversationGoal.LEAD,
        product="TINY",
        lead_intent=True,
    )
    routed = router.route(sales)
    assert routed.trace.get("should_retrieve") is True
    assert routed.conversation_goal == ConversationGoal.LEAD
    assert routed.current_turn_intent == TurnIntent.KNOWLEDGE


def test_ok_what_is_tiny_during_sales_keeps_goal_and_runs_rag() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    cid = "abc123"
    buy = orchestrator.handle(cid, "I want to buy TINY.")
    assert buy.conversation_goal == ConversationGoal.LEAD
    asked = orchestrator.handle(cid, "ok what is tiny")
    assert asked.conversation_goal == ConversationGoal.LEAD
    assert asked.product == "TINY"
    assert asked.current_turn_intent == TurnIntent.KNOWLEDGE
    assert asked.trace.get("should_retrieve") is True
    assert asked.query_rewritten == "what is TINY"
    assert knowledge.queries[-1] == "what is TINY"
    assert asked.trace.get("generation_temperature") == 0.0
    assert asked.trace.get("grounded") is True
    assert recorder.calls[-1]["temperature"] == 0.0
    assert asked.trace.get("state_before", {}).get("goal") == ConversationGoal.LEAD
    assert asked.trace.get("state_before", {}).get("product") == "TINY"
