from __future__ import annotations

from app.services.conversation.models import ConversationGoal, ConversationState, LeadStatus
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from tests.unit.conversation.fakes import FakeKnowledge, make_orchestrator
from tests.validation.harness import make_graph_stack


def _setup_lead_awaiting_phone(orchestrator, cid: str):
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Can someone call me?")
    state = orchestrator.handle(cid, "I'm Rahul from Delhi.")
    assert state.user_name == "Rahul"
    assert state.city == "Delhi"
    assert state.awaiting_field == "phone"
    assert state.lead_collection_active is True
    return state


def _phone_prompt_snippet() -> str:
    return "best number to reach you"


def test_01_lead_active_feature_question_resumes_in_same_response() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "resume-1"
    _setup_lead_awaiting_phone(orchestrator, cid)
    result = orchestrator.handle(cid, "What is the main feature of TINY?")
    assert knowledge.queries
    assert result.lead_collection_active is True
    assert result.conversation_goal == ConversationGoal.LEAD
    assert result.awaiting_field == "phone"
    assert result.user_name == "Rahul"
    assert result.city == "Delhi"
    assert result.product == "TINY"
    assert result.response != INSUFFICIENT_INFORMATION_MESSAGE
    assert _phone_prompt_snippet() in result.response.lower()
    assert "if you'd like help" in result.response.lower()
    assert "absolutely" not in result.response.lower()
    assert result.trace.get("lead_resume_appended") is True


def test_02_lead_active_warranty_question_resumes_in_same_response() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "resume-2"
    _setup_lead_awaiting_phone(orchestrator, cid)
    result = orchestrator.handle(cid, "What is TINY's warranty?")
    assert any("warranty" in query.lower() for query in knowledge.queries)
    assert _phone_prompt_snippet() in result.response.lower()
    assert result.awaiting_field == "phone"


def test_03_lead_active_price_question_resumes_in_same_response() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "resume-3"
    _setup_lead_awaiting_phone(orchestrator, cid)
    queries_before = len(knowledge.queries)
    result = orchestrator.handle(cid, "How much does TINY cost?")
    assert len(knowledge.queries) == queries_before
    assert "earkart.com" in result.response.lower()
    assert _phone_prompt_snippet() in result.response.lower()
    assert result.awaiting_field == "phone"


def test_04_lead_active_multiple_knowledge_questions_resume_once() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "resume-4"
    _setup_lead_awaiting_phone(orchestrator, cid)
    result = orchestrator.handle(
        cid,
        "What are TINY's features and what is its warranty?",
    )
    assert len(knowledge.queries) >= 2
    assert _phone_prompt_snippet() in result.response.lower()
    assert result.response.lower().count(_phone_prompt_snippet()) == 1
    assert result.awaiting_field == "phone"


def test_informal_know_more_during_phone_collection_retrieves() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "resume-informal"
    _setup_lead_awaiting_phone(orchestrator, cid)
    result = orchestrator.handle(cid, "i wanna know more in deatils of tiny")
    assert knowledge.queries
    assert result.lead_collection_active is True
    assert result.awaiting_field == "phone"
    assert result.conversation_goal == ConversationGoal.LEAD
    assert _phone_prompt_snippet() in result.response.lower()
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("resume-5", "What is the main feature of TINY?")
    assert knowledge.queries
    assert result.lead_collection_active is False
    assert _phone_prompt_snippet() not in result.response.lower()
    assert result.awaiting_field == ""


def test_06_ok_after_knowledge_is_acknowledgement_not_lead_resume_trigger() -> None:
    orchestrator, *_ = make_orchestrator()
    cid = "resume-6"
    _setup_lead_awaiting_phone(orchestrator, cid)
    knowledge_turn = orchestrator.handle(cid, "What is TINY's main feature?")
    assert _phone_prompt_snippet() in knowledge_turn.response.lower()
    ok_turn = orchestrator.handle(cid, "ok")
    assert ok_turn.awaiting_field == "phone"
    assert ok_turn.trace.get("acknowledgement_only") is True
    assert ok_turn.response.lower().count(_phone_prompt_snippet()) == 0


def test_07_lead_active_product_switch_preserves_lead_state() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "resume-7"
    _setup_lead_awaiting_phone(orchestrator, cid)
    result = orchestrator.handle(cid, "Actually tell me about Bluup.")
    assert any("bluup" in query.lower() for query in knowledge.queries)
    assert result.product == "TINY"
    assert result.user_name == "Rahul"
    assert result.city == "Delhi"
    assert result.lead_collection_active is True
    assert _phone_prompt_snippet() in result.response.lower()


def test_08_phone_after_knowledge_response_is_stored_without_duplicate_ask() -> None:
    orchestrator, _knowledge, _llm, lead_tool, *_ = make_orchestrator()
    cid = "resume-8"
    _setup_lead_awaiting_phone(orchestrator, cid)
    knowledge_turn = orchestrator.handle(cid, "What is TINY's main feature?")
    assert _phone_prompt_snippet() in knowledge_turn.response.lower()
    phone_turn = orchestrator.handle(cid, "+91 9876543210")
    assert phone_turn.phone == "+919876543210"
    assert phone_turn.awaiting_field != "phone"
    assert _phone_prompt_snippet() not in phone_turn.response.lower()
    assert lead_tool.leads


def test_acceptance_connect_name_details_then_phone() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    cid = "acceptance-resume"
    orchestrator.handle(cid, "yes connect me with the sales team")
    orchestrator.handle(cid, "sudhanshu")
    result = orchestrator.handle(cid, "give me details of tiny")
    assert knowledge.queries
    assert _phone_prompt_snippet() in result.response.lower()
    assert "if you'd like help" in result.response.lower()
    assert "absolutely" not in result.response.lower()
    assert result.awaiting_field == "phone"
    assert result.lead_collection_active is True
    assert result.trace.get("lead_resume_appended") is True
    phone_turn = orchestrator.handle(cid, "9876543210")
    assert phone_turn.phone == "+919876543210"
    assert phone_turn.phone_country == "IN"
    assert "which city" in phone_turn.response.lower()
    assert "absolutely" not in phone_turn.response.lower()


def test_resume_when_awaiting_field_without_collection_flag() -> None:
    from app.helpers.workflow_resume import append_workflow_resume_after_knowledge

    state = ConversationState(
        conversation_goal=ConversationGoal.LEAD,
        user_name="sudhanshu",
        awaiting_field="phone",
        lead_collection_active=False,
        trace={"next_action": "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD", "resume_lead_after_knowledge": True},
    )
    composed = append_workflow_resume_after_knowledge(state, "TINY is rechargeable.")
    assert _phone_prompt_snippet() in composed.lower()
    assert "if you'd like help" in composed.lower()
    assert "absolutely" not in composed.lower()
    assert state.lead_collection_active is True


def test_support_ticket_collection_resumes_after_knowledge() -> None:
    knowledge = FakeKnowledge()
    orchestrator, _, _, tickets, *_ = make_graph_stack(knowledge=knowledge)
    cid = "resume-support"
    orchestrator.handle(cid, "My Radius M16 isn't working.")
    orchestrator.handle(cid, "Can you create a support ticket?")
    orchestrator.handle(cid, "Rahul")
    result = orchestrator.handle(cid, "What is BTE?")
    assert knowledge.queries[-1] == "What is BTE?"
    assert result.conversation_goal == ConversationGoal.SUPPORT
    assert "our team to reach you" in result.response.lower()
    assert tickets.tickets == []
