"""Regression tests for hearing-loss consultation and product-interest quick replies."""

import json

from app.helpers.conversation_extract import (
    looks_like_acquisition_intent,
    looks_like_hearing_consultation_need,
    looks_like_product_interest_request,
)
from app.helpers.quick_replies import (
    CONSULTATION_TRIAL_LABEL,
    CONSULTATION_TRIAL_MESSAGE,
    quick_replies_from_trace,
)
from app.services.conversation.models import ChatMode, ConversationGoal, TurnIntent
from tests.unit.conversation.fakes import RecordingLLM, make_orchestrator
from tests.unit.conversation.test_semantic_router import ScriptedRouterLLM

HEARING_LOSS_QUERY = "i have hearing loss of 45% percent which hearing aid should i need"
HEARING_LOSS_REPLY = (
    "Your hearing loss of 45% requires a hearing aid that is tailored to your specific needs. "
    "An audiologist can assess your hearing and recommend the most suitable device. "
    "Would you like to know more about the types of hearing aids available or how to find one "
    "that fits your lifestyle?"
)
PRODUCT_INTEREST = "I want a hearing aid"
PRODUCT_INTEREST_BUTTON = f"followup:{PRODUCT_INTEREST}"


def _support_semantic_llm() -> ScriptedRouterLLM:
    payload = json.dumps(
        {
            "intent": "support",
            "action": "start_support",
            "needs_rewrite": False,
            "product": None,
            "confidence": 0.95,
        }
    )
    return ScriptedRouterLLM([payload, payload])


def test_hearing_loss_selection_question_is_consultation_not_support() -> None:
    assert looks_like_hearing_consultation_need(HEARING_LOSS_QUERY)
    assert looks_like_product_interest_request(PRODUCT_INTEREST)


def test_hinglish_hearing_aid_request_is_acquisition_not_support() -> None:
    for message in (
        "mujhe hearing aid chaiye",
        "hi mujhe ek hearing aid chaiye",
        "mujhe ek hearing aid chahiye",
        "hearing aid chahiye",
        "mujhe hearing chaiye",
        "i need hearing aid",
        "मुझे हियरिंग एड चाहिए",
        "hi mujhe heaing aid chaiye thi",
        "hi mujhe heaing Aid chaiye",
        "i want to buy heaing aid",
    ):
        assert looks_like_acquisition_intent(message)
    assert not looks_like_acquisition_intent("my hearing aid is not working")


def test_hearing_loss_query_routes_to_knowledge() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("hl-knowledge", HEARING_LOSS_QUERY)
    assert result.conversation_goal == ConversationGoal.KNOWLEDGE
    assert result.mode == ChatMode.KNOWLEDGE
    assert result.current_turn_intent == TurnIntent.KNOWLEDGE
    assert result.conversation_goal != ConversationGoal.SUPPORT
    assert knowledge.queries


def test_hearing_loss_answer_offers_product_interest_buttons() -> None:
    answer = json.dumps({"grounded": True, "answer": HEARING_LOSS_REPLY, "source_ids": ["chunk-1"]})
    orchestrator, *_ = make_orchestrator(llm=RecordingLLM(output=answer))
    result = orchestrator.handle("hl-buttons", HEARING_LOSS_QUERY)
    replies = quick_replies_from_trace(result)
    labels = [item["label"] for item in replies]
    assert "I want a hearing aid" in labels
    assert CONSULTATION_TRIAL_LABEL in labels


def test_hinglish_hearing_aid_request_routes_to_lead_even_when_semantic_says_support() -> None:
    orchestrator, *_ = make_orchestrator(llm=_support_semantic_llm())
    result = orchestrator.handle("hinglish-acquisition", "mujhe hearing aid chaiye")
    assert result.conversation_goal != ConversationGoal.SUPPORT
    assert result.mode != ChatMode.SUPPORT
    assert result.support_intent is False
    assert result.current_turn_intent == TurnIntent.LEAD_INTENT
    assert "having trouble" not in (result.response or "").lower()


def test_typo_hearing_aid_requests_route_to_lead_not_knowledge() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("typo-acquisition", "hi mujhe heaing aid chaiye thi")
    assert result.current_turn_intent == TurnIntent.LEAD_INTENT
    assert result.mode == ChatMode.LEAD
    assert knowledge.queries == []
    assert "hearing aid options" not in (result.response or "").lower()


def test_hindi_and_shorthand_hearing_requests_route_to_lead() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    hindi = orchestrator.handle("hi-acquisition", "मुझे हियरिंग एड चाहिए")
    assert hindi.current_turn_intent == TurnIntent.LEAD_INTENT
    assert hindi.mode == ChatMode.LEAD
    assert hindi.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES}
    assert knowledge.queries == []
    assert "ear health" not in (hindi.response or "").lower()

    shorthand = orchestrator.handle("hinglish-short", "mujhe hearing chaiye")
    assert shorthand.current_turn_intent == TurnIntent.LEAD_INTENT
    assert shorthand.mode == ChatMode.LEAD
    assert shorthand.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES}
    assert knowledge.queries == []
    assert "jankari" not in (shorthand.response or "").lower()


def test_i_want_a_hearing_aid_after_hearing_loss_is_sales_not_support() -> None:
    orchestrator, *_ = make_orchestrator(llm=_support_semantic_llm())
    cid = "hl-sales-not-support"
    orchestrator.handle(cid, HEARING_LOSS_QUERY)
    second = orchestrator.handle(cid, PRODUCT_INTEREST)
    assert second.conversation_goal != ConversationGoal.SUPPORT
    assert second.mode != ChatMode.SUPPORT
    assert second.support_intent is False
    assert second.current_turn_intent == TurnIntent.LEAD_INTENT
    assert "customer service team" not in (second.response or "").lower()
    assert "having trouble" not in (second.response or "").lower()


def test_i_want_a_hearing_aid_button_click_after_hearing_loss_is_sales_not_support() -> None:
    answer = json.dumps({"grounded": True, "answer": HEARING_LOSS_REPLY, "source_ids": ["chunk-1"]})
    orchestrator, *_ = make_orchestrator(llm=_support_semantic_llm_with_generation(answer))
    cid = "hl-button-click"
    first = orchestrator.handle(cid, HEARING_LOSS_QUERY)
    assert quick_replies_from_trace(first)
    second = orchestrator.handle(cid, PRODUCT_INTEREST_BUTTON)
    assert second.user_message == PRODUCT_INTEREST
    assert second.conversation_goal != ConversationGoal.SUPPORT
    assert second.mode != ChatMode.SUPPORT
    assert second.support_intent is False
    assert "customer service team" not in (second.response or "").lower()
    assert "having trouble" not in (second.response or "").lower()


def test_create_trial_after_hearing_loss_starts_lead_collection() -> None:
    orchestrator, _, _, lead_tool, *_ = make_orchestrator(llm=_support_semantic_llm())
    cid = "hl-trial-lead"
    orchestrator.handle(cid, HEARING_LOSS_QUERY)
    second = orchestrator.handle(cid, CONSULTATION_TRIAL_MESSAGE)
    assert second.explicit_action == "create_lead" or second.lead_collection_active
    assert second.awaiting_field in {"name", "phone", "phone_country", "city"}
    assert second.conversation_goal != ConversationGoal.SUPPORT
    assert lead_tool.leads == []


def _support_semantic_llm_with_generation(generation_answer: str) -> ScriptedRouterLLM:
    support_payload = json.dumps(
        {
            "intent": "support",
            "action": "start_support",
            "needs_rewrite": False,
            "product": None,
            "confidence": 0.95,
        }
    )
    return ScriptedRouterLLM([generation_answer, support_payload, generation_answer])
