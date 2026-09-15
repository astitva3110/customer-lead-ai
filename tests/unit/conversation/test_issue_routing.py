"""Regression tests for general hearing vs device support and explicit ticket/lead actions."""

from __future__ import annotations

from app.helpers.conversation_extract import looks_like_ticket_request
from app.services.conversation.models import ConversationGoal, TicketStatus, TurnIntent
from tests.unit.conversation.fakes import FakeKnowledge, make_orchestrator


def test_general_hearing_questions_use_rag_not_support_ticket() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    messages = (
        "i have hearing loss",
        "I have hearing problem",
        "I have hearing problem what should I do",
        "i have hearing issues",
        "I can't hear well",
    )
    for message in messages:
        cid = f"hearing-{hash(message) & 0xffff}"
        result = orchestrator.handle(cid, message)
        assert result.conversation_goal != ConversationGoal.SUPPORT, message
        assert result.current_turn_intent == TurnIntent.KNOWLEDGE, message
        assert result.trace.get("should_retrieve") is True, message
        assert "support ticket" not in (result.response or "").lower(), message


def test_device_fault_still_offers_support_ticket() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    result = orchestrator.handle("device-fault", "my Radius is not working")
    assert result.conversation_goal == ConversationGoal.SUPPORT
    assert result.current_turn_intent == TurnIntent.SUPPORT_INTENT
    assert "our team" in (result.response or "").lower()
    assert not knowledge.queries


def test_explicit_create_ticket_starts_collection_not_offer() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    for message in (
        "please create a support ticket",
        "create a ticket",
        "okay create a ticket",
        "create ticket",
    ):
        assert looks_like_ticket_request(message), message
        cid = f"ticket-{hash(message) & 0xffff}"
        result = orchestrator.handle(cid, message)
        assert result.explicit_action == "create_ticket", message
        assert result.awaiting_field in {"name", "phone", "phone_country", "product", "issue"}, message
        assert "would you like me to do that" not in (result.response or "").lower(), message
        assert not knowledge.queries


def test_explicit_create_lead_starts_collection_not_offer() -> None:
    orchestrator, knowledge, *_ = make_orchestrator(knowledge=FakeKnowledge())
    result = orchestrator.handle("lead-explicit", "please create a lead")
    assert result.explicit_action == "create_lead"
    assert result.awaiting_field in {"name", "phone", "phone_country", "city"}
    assert "would you like me to do that" not in (result.response or "").lower()
    assert not knowledge.queries


def test_explicit_create_ticket_skips_known_whatsapp_identity_fields() -> None:
    orchestrator, knowledge, _, _, ticket_tool, _ = make_orchestrator(knowledge=FakeKnowledge())
    first = orchestrator.handle(
        None,
        "My TINY is not working.",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
        user_name="Rahul",
    )
    assert first.conversation_goal == ConversationGoal.SUPPORT

    ticket = orchestrator.handle(
        first.conversation_id,
        "create ticket",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
        user_name="Rahul",
    )
    assert ticket.explicit_action == "create_ticket"
    assert ticket.awaiting_field in {"product", "issue", ""}
    assert "name should" not in (ticket.response or "").lower()
    assert "number" not in (ticket.response or "").lower()
    if ticket.ticket_status == TicketStatus.CREATED:
        assert ticket_tool.tickets[0].name == "Rahul"
        assert ticket_tool.tickets[0].phone == "+919876543210"


def test_yes_after_rag_battery_followup_routes_to_knowledge_not_support() -> None:
    from app.services.conversation.models import ChatMode, ConversationState
    from app.services.conversation.router import ChatRouter

    battery_followup = (
        "You may be able to use a hearing aid, but the appropriate device depends on "
        "your actual hearing-test results and individual needs. An audiologist can recommend "
        "whether amplification would be beneficial. Would you like to know about its battery life?"
    )
    router = ChatRouter()
    routed = router.route(
        ConversationState(
            user_message="yes",
            conversation_goal=ConversationGoal.KNOWLEDGE,
            mode=ChatMode.KNOWLEDGE,
            support_intent=True,
            conversation_history=[
                {"role": "user", "content": "i have 30 percent in one ear hearing loss"},
                {"role": "assistant", "content": battery_followup},
            ],
        )
    )
    assert routed.current_turn_intent == TurnIntent.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.support_intent is False
