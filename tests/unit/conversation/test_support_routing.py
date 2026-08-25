from __future__ import annotations

from app.helpers.conversation_extract import (
    looks_like_support_contact_request,
    looks_like_support_escalation_request,
)
from app.helpers.conversation_reply import repeats_recent_assistant
from app.helpers.conversation_turn import is_short_yes
from app.services.conversation.models import (
    ChatMode,
    ConversationGoal,
    ConversationState,
    TicketStatus,
    TurnIntent,
)
from app.services.conversation.router import ChatRouter
from tests.unit.conversation.fakes import FakeKnowledge, make_orchestrator


def test_support_contact_phrases_detected() -> None:
    assert looks_like_support_contact_request("i want to talk to your customer support")
    assert looks_like_support_contact_request("i want to contact your customer service")
    assert looks_like_support_contact_request("i need to talk your support team")
    assert not looks_like_support_contact_request("I want to buy TINY.")


def test_support_escalation_phrases_detected() -> None:
    assert looks_like_support_escalation_request("i want to repair")
    assert looks_like_support_escalation_request("i need to reapir")
    assert looks_like_support_escalation_request("i need assitance")
    assert looks_like_support_escalation_request("i need assistance")
    assert not looks_like_support_escalation_request("I want to buy TINY.")


def test_yes_plz_is_short_yes() -> None:
    assert is_short_yes("yes plz")
    assert is_short_yes("yeah please")


def test_yeah_in_support_stays_support_not_knowledge() -> None:
    router = ChatRouter()
    state = ConversationState(
        user_message="yeah",
        product="TINY",
        conversation_goal=ConversationGoal.SUPPORT,
        mode=ChatMode.SUPPORT,
        support_issue="it is not working",
        conversation_history=[
            {"role": "user", "content": "my TINY is not working"},
            {
                "role": "assistant",
                "content": "I'm sorry you're dealing with that on TINY. I'll help you work through it.",
            },
        ],
    )
    routed = router.route(state)
    assert routed.current_turn_intent != TurnIntent.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is not True
    assert routed.conversation_goal == ConversationGoal.SUPPORT


def test_yes_plz_in_support_is_confirmation() -> None:
    router = ChatRouter()
    state = ConversationState(
        user_message="yes plz",
        product="TINY",
        conversation_goal=ConversationGoal.SUPPORT,
        mode=ChatMode.SUPPORT,
        support_issue="it is not working",
        conversation_history=[
            {"role": "user", "content": "my TINY is not working"},
            {"role": "assistant", "content": "Would you like troubleshooting help?"},
        ],
    )
    routed = router.route(state)
    assert routed.current_turn_intent == TurnIntent.CONFIRMATION
    assert routed.trace.get("should_retrieve") is not True


def test_yeah_after_buy_still_routes_to_knowledge_for_lead() -> None:
    router = ChatRouter()
    state = ConversationState(
        user_message="yes",
        product="TINY",
        conversation_goal=ConversationGoal.SALES,
        mode=ChatMode.LEAD,
        conversation_history=[
            {"role": "user", "content": "I want to buy TINY."},
            {"role": "assistant", "content": "Absolutely! TINY is a great choice."},
        ],
    )
    routed = router.route(state)
    assert routed.current_turn_intent == TurnIntent.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True


def test_support_contact_starts_ticket_collection() -> None:
    router = ChatRouter()
    state = ConversationState(
        user_message="i want to contact your customer service",
        product="TINY",
        conversation_goal=ConversationGoal.SUPPORT,
        mode=ChatMode.SUPPORT,
        support_issue="it is not working",
    )
    routed = router.route(state)
    assert routed.explicit_action == "create_ticket"
    assert routed.mode == ChatMode.SUPPORT
    assert routed.conversation_goal == ConversationGoal.SUPPORT


def test_repair_in_support_starts_ticket_collection() -> None:
    router = ChatRouter()
    state = ConversationState(
        user_message="i want to repair",
        product="TINY",
        conversation_goal=ConversationGoal.SUPPORT,
        mode=ChatMode.SUPPORT,
        support_issue="it is not working",
    )
    routed = router.route(state)
    assert routed.explicit_action == "create_ticket"
    assert routed.ticket_status == TicketStatus.COLLECTING


def test_need_help_on_lead_does_not_open_ticket() -> None:
    router = ChatRouter()
    routed = router.route(ConversationState(user_message="I want to buy TINY and need help choosing"))
    assert routed.mode == ChatMode.LEAD
    assert routed.explicit_action != "create_ticket"


def test_assistance_typo_in_support_starts_ticket_collection() -> None:
    router = ChatRouter()
    state = ConversationState(
        user_message="i need assitance",
        product="TINY",
        conversation_goal=ConversationGoal.SUPPORT,
        mode=ChatMode.SUPPORT,
        support_issue="it is not working",
    )
    routed = router.route(state)
    assert routed.explicit_action == "create_ticket"


def test_repeats_recent_assistant_catches_earlier_duplicate() -> None:
    state = ConversationState(
        conversation_history=[
            {"role": "assistant", "content": "I'm sorry to hear that your hearing aid Tiny is not working."},
            {"role": "assistant", "content": "I'm sorry you're dealing with that on TINY."},
        ]
    )
    assert repeats_recent_assistant(
        state,
        "I'm sorry to hear that your hearing aid Tiny is not working.",
    )


def test_trace_regression_support_conversation() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    cid = "support-trace-1"

    first = orchestrator.handle(cid, "i had buy tiny from U and it is not working")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.product == "TINY"
    assert first.support_issue
    assert not knowledge.queries

    second = orchestrator.handle(cid, "i do not know about that what not working")
    assert second.conversation_goal == ConversationGoal.SUPPORT
    assert not knowledge.queries

    third = orchestrator.handle(cid, "yeah")
    assert third.conversation_goal == ConversationGoal.SUPPORT
    assert third.trace.get("should_retrieve") is not True
    assert knowledge.queries == []

    fourth = orchestrator.handle(cid, "i want to talk to your custommer support")
    assert fourth.explicit_action == "create_ticket"
    assert fourth.ticket_status == TicketStatus.COLLECTING
    assert fourth.awaiting_field in {"name", "phone", "product", "issue"}

    fifth = orchestrator.handle(cid, "i want to contact your customer service")
    assert fifth.conversation_goal == ConversationGoal.SUPPORT
    assert fifth.explicit_action == "create_ticket"


def test_repair_trace_regression() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    cid = "support-trace-2"

    first = orchestrator.handle(cid, "i hearing Aid tiny is not working")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.product == "TINY"
    assert not knowledge.queries

    second = orchestrator.handle(cid, "i want to repair")
    assert second.explicit_action == "create_ticket"
    assert second.ticket_status == TicketStatus.COLLECTING
    assert second.awaiting_field in {"name", "phone", "product", "issue"}
    assert not knowledge.queries

    third = orchestrator.handle(cid, "yes plz")
    assert third.conversation_goal == ConversationGoal.SUPPORT
    assert third.current_turn_intent == TurnIntent.CONFIRMATION
    assert third.trace.get("should_retrieve") is not True

    fourth = orchestrator.handle(cid, "i need assitance")
    assert fourth.conversation_goal == ConversationGoal.SUPPORT
    assert fourth.explicit_action == "create_ticket"


def test_reapir_typo_routes_to_support_not_knowledge() -> None:
    router = ChatRouter()
    routed = router.route(ConversationState(user_message="i need to reapir my hearing Aid"))
    assert routed.current_turn_intent == TurnIntent.SUPPORT_INTENT
    assert routed.explicit_action == "create_ticket"
    assert routed.trace.get("should_retrieve") is not True


def test_trace_regression_8cdae2e9_conversation() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    cid = "support-trace-8cdae2e9"

    first = orchestrator.handle(cid, "i need to reapir my hearing Aid")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.explicit_action == "create_ticket"
    assert first.awaiting_field in {"name", "phone", "product", "issue"}
    assert not knowledge.queries

    second = orchestrator.handle(cid, "i have tiny it is not working")
    assert second.conversation_goal == ConversationGoal.SUPPORT
    assert second.product == "TINY"
    assert not knowledge.queries

    third = orchestrator.handle(cid, "i need assistance")
    assert third.conversation_goal == ConversationGoal.SUPPORT
    assert third.explicit_action == "create_ticket"
    assert third.awaiting_field in {"name", "phone", "product", "issue"}

    fourth = orchestrator.handle(cid, "i need to repair it")
    assert fourth.conversation_goal == ConversationGoal.SUPPORT
    assert fourth.explicit_action == "create_ticket"
    assert fourth.awaiting_field in {"name", "phone", "product", "issue"}
    assert not knowledge.queries


def test_trace_regression_hearing_not_working_conversation() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    cid = "support-trace-3"

    first = orchestrator.handle(cid, "my hearing is not working")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.support_issue
    assert first.explicit_action != "create_ticket"
    assert not knowledge.queries

    second = orchestrator.handle(cid, "i want to repair it")
    assert second.conversation_goal == ConversationGoal.SUPPORT
    assert second.explicit_action == "create_ticket"
    assert second.awaiting_field in {"name", "phone", "product", "issue"}
    assert not knowledge.queries

    third = orchestrator.handle(cid, "i want to talk to your customer support")
    assert third.explicit_action == "create_ticket"
    assert third.ticket_status == TicketStatus.COLLECTING

    fourth = orchestrator.handle(cid, "i need to talk your support team")
    assert fourth.explicit_action == "create_ticket"
    assert fourth.conversation_goal == ConversationGoal.SUPPORT

    fifth = orchestrator.handle(cid, "ok")
    assert fifth.conversation_goal == ConversationGoal.SUPPORT
    assert fifth.current_turn_intent == TurnIntent.CONFIRMATION
    assert fifth.trace.get("should_retrieve") is not True
