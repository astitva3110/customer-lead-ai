from app.helpers.conversation_extract import (
    looks_like_device_support_issue,
    looks_like_general_hearing_concern,
    looks_like_hearing_consultation_need,
)
from app.helpers.hearing_symptom_normalize import (
    HearingSymptomTopic,
    analyze_hearing_symptom,
    canonical_hearing_query,
)
from app.services.conversation.models import ChatMode, ConversationGoal, ConversationState, TurnIntent
from app.services.conversation.query_rewriter import QueryRewriter
from app.services.conversation.router import ChatRouter


def test_awaz_nahi_is_personal_hearing_not_device() -> None:
    for message in (
        "mujhe awaz nhi aa rhe hai",
        "mujhe awaz nahi aa rahe hai",
        "mujhe awaz nahi aati",
        "sunai nahi deta",
        "sunai problem hore",
        "1aaankh sunai problem hore",
    ):
        analysis = analyze_hearing_symptom(message)
        assert analysis.matched
        assert analysis.topic == HearingSymptomTopic.HEARING_DIFFICULTY
        assert analysis.device_mentioned is False
        assert looks_like_general_hearing_concern(message)
        assert looks_like_hearing_consultation_need(message)
        assert not looks_like_device_support_issue(message)


def test_hearing_aid_awaz_issue_is_device_support() -> None:
    for message in (
        "mera hearing aid se awaz nahi aa rahi",
        "TINY se awaz nahi aa rahi",
        "my hearing aid has no sound",
    ):
        analysis = analyze_hearing_symptom(message)
        assert analysis.matched
        assert analysis.topic in {
            HearingSymptomTopic.DEVICE_NO_SOUND,
            HearingSymptomTopic.DEVICE_LOW_VOLUME,
            HearingSymptomTopic.DEVICE_FAULT,
        }
        assert looks_like_device_support_issue(message)


def test_canonical_query_is_english_for_retrieval() -> None:
    assert canonical_hearing_query("mujhe awaz nahi aa rahe hai") == (
        "difficulty hearing or cannot hear sound"
    )


def test_query_rewriter_normalizes_hearing_symptom_before_routing() -> None:
    state = ConversationState(user_message="mujhe awaz nahi aa rahe hai")
    result = QueryRewriter().normalize(state)
    assert result.rewritten_query.lower() == "difficulty hearing or cannot hear sound"
    assert state.trace["hearing_symptom"]["topic"] == "hearing_difficulty"


def test_informational_hearing_loss_question_is_not_personal_symptom() -> None:
    assert not looks_like_general_hearing_concern("What is hearing loss?")


def test_awaz_symptom_routes_to_knowledge_not_support() -> None:
    state = ConversationState(user_message="mujhe awaz nahi aa rahe hai")
    QueryRewriter().normalize(state)
    routed = ChatRouter().route(state)
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.current_turn_intent == TurnIntent.KNOWLEDGE
    assert routed.conversation_goal != ConversationGoal.SUPPORT
