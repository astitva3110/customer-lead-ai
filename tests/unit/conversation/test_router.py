from app.services.conversation.models import ChatMode, ConversationGoal, ConversationState, LeadStatus
from app.services.conversation.router import ChatRouter


def test_knowledge_question_routes_to_knowledge() -> None:
    router = ChatRouter()
    state = router.route(ConversationState(user_message="What is Radius M16?"))
    assert state.mode == ChatMode.KNOWLEDGE
    assert state.intent == ChatMode.KNOWLEDGE
    for message in (
        "Why should I buy from Earkart?",
        "What are the benefits of buying from Earkart?",
        "What should I buy from Earkart?",
    ):
        assert router.route(ConversationState(user_message=message)).mode == ChatMode.KNOWLEDGE


def test_purchase_intent_routes_to_lead() -> None:
    router = ChatRouter()
    assert router.route(ConversationState(user_message="I want to buy Radius M16.")).mode == ChatMode.LEAD
    for message in (
        "I want to buy TINY.",
        "I want to purchase Radius M16.",
        "Can I buy TINY?",
    ):
        assert router.route(ConversationState(user_message=message)).mode == ChatMode.LEAD


def test_interested_in_buying_routes_to_lead() -> None:
    state = ChatRouter().route(ConversationState(user_message="I am interested in buying this."))
    assert state.mode == ChatMode.LEAD


def test_support_issue_routes_to_support() -> None:
    state = ChatRouter().route(ConversationState(user_message="My hearing aid is not working."))
    assert state.mode == ChatMode.SUPPORT


def test_stopped_working_routes_to_support() -> None:
    state = ChatRouter().route(ConversationState(user_message="My Radius M16 stopped working."))
    assert state.mode == ChatMode.SUPPORT


def test_phone_number_stays_in_lead_without_reroute() -> None:
    state = ConversationState(
        user_message="+91 9876543210",
        mode=ChatMode.LEAD,
        awaiting_field="phone",
        lead_status=LeadStatus.COLLECTING,
    )
    routed = ChatRouter().route(state)
    assert routed.mode == ChatMode.LEAD
    assert routed.intent == ChatMode.LEAD


def test_knowledge_question_during_support_sets_return_mode() -> None:
    state = ConversationState(
        user_message="Actually, how long is the warranty?",
        mode=ChatMode.SUPPORT,
        awaiting_field="name",
    )
    routed = ChatRouter().route(state)
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.return_mode == ChatMode.SUPPORT


def test_tell_me_about_during_lead_retrieves_and_keeps_goal() -> None:
    state = ConversationState(
        user_message="Actually, tell me about BTE first.",
        mode=ChatMode.LEAD,
        conversation_goal=ConversationGoal.LEAD,
        product="TINY",
        lead_intent=True,
        lead_status=LeadStatus.COLLECTING,
    )
    routed = ChatRouter().route(state)
    assert routed.trace.get("should_retrieve") is True
    assert routed.current_turn_intent == "KNOWLEDGE"
    assert routed.conversation_goal == ConversationGoal.LEAD
    assert routed.return_mode == ChatMode.LEAD
    assert routed.product == "TINY"


def test_how_are_you_greeting_does_not_retrieve() -> None:
    routed = ChatRouter().route(ConversationState(user_message="Hi, how are you?"))
    assert routed.trace.get("should_retrieve") is False
    assert routed.trace.get("needs_natural_reply") is True
    assert routed.current_turn_intent == "GENERAL"


def test_greeting_plus_product_question_retrieves() -> None:
    routed = ChatRouter().route(ConversationState(user_message="Hi, I want to know about Bluup."))
    assert routed.trace.get("should_retrieve") is True
    assert routed.current_turn_intent == "KNOWLEDGE"
