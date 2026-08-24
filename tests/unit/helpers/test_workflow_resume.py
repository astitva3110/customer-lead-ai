from app.helpers.workflow_resume import (
    append_workflow_resume_after_knowledge,
    next_missing_lead_field,
    should_resume_lead_after_knowledge,
)
from app.services.conversation.models import ConversationGoal, ConversationState


def test_next_missing_lead_field_respects_priority() -> None:
    state = ConversationState(user_name="Rahul", city="Delhi")
    assert next_missing_lead_field(state) == "phone"


def test_should_resume_lead_when_collection_active() -> None:
    state = ConversationState(
        lead_collection_active=True,
        conversation_goal=ConversationGoal.SALES,
        trace={"next_action": "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD"},
    )
    assert should_resume_lead_after_knowledge(state) is True


def test_should_resume_lead_when_awaiting_field_even_without_flag() -> None:
    state = ConversationState(
        conversation_goal=ConversationGoal.LEAD,
        user_name="sudhanshu",
        awaiting_field="phone",
        lead_collection_active=False,
        trace={"next_action": "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD", "resume_lead_after_knowledge": True},
    )
    assert should_resume_lead_after_knowledge(state) is True


def test_append_lead_resume_composes_single_message() -> None:
    state = ConversationState(
        user_name="Rahul",
        city="Delhi",
        lead_collection_active=True,
        conversation_goal=ConversationGoal.SALES,
        trace={"resume_lead_after_knowledge": True},
    )
    composed = append_workflow_resume_after_knowledge(state, "TINY is rechargeable.")
    assert "TINY is rechargeable." in composed
    assert "best number to reach you" in composed.lower()
    assert "if you'd like help" in composed.lower()
    assert "absolutely" not in composed.lower()
    assert state.awaiting_field == "phone"
    assert state.trace.get("lead_resume_appended") is True
