from __future__ import annotations

from app.services.conversation.models import ConversationGoal, ConversationState, LeadStatus, LeadWorkflow
from app.helpers.conversation_prompt import CONVERSATION_SYSTEM_PROMPT, build_conversation_user_prompt


def test_system_prompt_forbids_form_script_and_forced_questions() -> None:
    text = CONVERSATION_SYSTEM_PROMPT.lower()
    assert "do not always end with a question" in text
    assert "form" in text
    assert "awaiting_phone" in text
    assert "lead" in text
    assert "do not describe the company" in text or "do not introduce the company" in text
    assert "do not use repetitive closers" in text


def test_user_prompt_hides_internal_workflow() -> None:
    state = ConversationState(
        user_message="I want to buy TINY.",
        conversation_goal=ConversationGoal.LEAD,
        lead_status=LeadStatus.COLLECTING,
        lead_workflow=LeadWorkflow.COLLECTING_PHONE,
        awaiting_field="phone",
        product="TINY",
        trace={"capabilities": ["CONVERSATION_ONLY"]},
    )
    prompt = build_conversation_user_prompt(state)
    assert "conversation_goal" not in prompt
    assert "LEAD" not in prompt
    assert "SUPPORT" not in prompt
    assert "awaiting_field" not in prompt
    assert "awaiting_phone" not in prompt
    assert "COLLECTING_PHONE" not in prompt
    assert "missing_lead_fields" not in prompt
    assert "current_turn_intent" not in prompt
    assert "considering a purchase" in prompt
    assert "TINY" in prompt


def test_greeting_prompt_does_not_pitch_the_company() -> None:
    prompt = build_conversation_user_prompt(ConversationState(user_message="Hi, how are you?"))
    assert "greeted" in prompt.lower()
    assert "do not describe the company" in prompt.lower()
    assert "LEAD" not in prompt
    assert "SUPPORT" not in prompt


def test_name_is_available_without_asking_again() -> None:
    prompt = build_conversation_user_prompt(
        ConversationState(
            user_message="My name is Rahul.",
            user_name="Rahul",
            conversation_goal=ConversationGoal.LEAD,
            current_turn_intent="CONTEXT_UPDATE",
            product="TINY",
        )
    )
    assert '"name": "Rahul"' in prompt
    assert "name_known" in prompt
    assert "awaiting_field" not in prompt
    assert "What name should we use" not in prompt
