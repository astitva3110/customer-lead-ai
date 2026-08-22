from __future__ import annotations

from app.services.conversation.models import ChatMode, ConversationGoal, LeadStatus, TurnIntent
from app.services.conversation.query_rewriter import QueryRewriter, bind_product_query, should_bind_product
from tests.unit.phase22.test_conversational_agent import _stack


def test_conversation_a_buy_then_knowledge_then_callback() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-2-a"
    buy = orchestrator.handle(cid, "bro i want to buy it tiny")
    assert buy.conversation_goal == ConversationGoal.LEAD
    assert buy.product == "TINY"
    assert buy.awaiting_field == ""
    assert lead.leads == []
    assert "phone" not in buy.response.lower()
    assert "what would you like to know" not in buy.response.lower()

    price = orchestrator.handle(cid, "what is the price?")
    assert price.conversation_goal == ConversationGoal.LEAD
    assert price.current_turn_intent == TurnIntent.KNOWLEDGE
    assert price.trace.get("should_retrieve") is True
    assert "TINY" in knowledge.queries[-1]
    assert "price" in knowledge.queries[-1].lower()

    warranty = orchestrator.handle(cid, "what about warranty?")
    assert warranty.conversation_goal == ConversationGoal.LEAD
    assert any("warranty" in query.lower() for query in knowledge.queries)
    assert lead.leads == []

    callback = orchestrator.handle(cid, "okay call me")
    assert callback.conversation_goal == ConversationGoal.LEAD
    assert callback.awaiting_field == "name"
    assert lead.leads == []


def test_conversation_b_support_then_knowledge_then_ticket() -> None:
    orchestrator, _, _, tickets, *_rest, knowledge = _stack()
    cid = "p22-2-b"
    first = orchestrator.handle(cid, "My hearing aid is giving very low sound.")
    assert first.conversation_goal == ConversationGoal.SUPPORT
    assert first.awaiting_field == ""
    assert "what name" not in first.response.lower()
    assert tickets.tickets == []

    named = orchestrator.handle(cid, "It is Radius M16.")
    assert named.product == "Radius M16"
    assert named.conversation_goal == ConversationGoal.SUPPORT

    check = orchestrator.handle(cid, "what should I check first?")
    assert check.conversation_goal == ConversationGoal.SUPPORT
    assert knowledge.queries
    assert check.trace.get("should_retrieve") is True

    ticket = orchestrator.handle(cid, "okay create a ticket")
    assert ticket.conversation_goal == ConversationGoal.SUPPORT
    assert ticket.awaiting_field
    assert tickets.tickets == []


def test_conversation_c_product_context_then_overwrite() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-2-c"
    orchestrator.handle(cid, "I want TINY.")
    asked = orchestrator.handle(cid, "Actually what is BTE?")
    assert asked.conversation_goal == ConversationGoal.LEAD
    assert asked.product == "TINY"
    assert "BTE" in knowledge.queries[-1] or "bte" in knowledge.queries[-1].lower()

    changed = orchestrator.handle(cid, "Actually I want Radius M16 instead.")
    assert changed.product == "Radius M16"
    assert changed.conversation_goal == ConversationGoal.LEAD

    callback = orchestrator.handle(cid, "Can someone call me?")
    assert callback.conversation_goal == ConversationGoal.LEAD
    assert callback.product == "Radius M16"
    assert callback.awaiting_field == "name"
    assert lead.leads == []


def test_conversation_d_greeting_plus_tiny_stays_grounded() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    result = orchestrator.handle("p22-2-d", "hi, what is TINY?")
    assert result.conversation_goal == ConversationGoal.KNOWLEDGE
    assert result.mode == ChatMode.KNOWLEDGE
    assert knowledge.queries
    assert result.sources
    assert recorder.calls[-1]["temperature"] == 0.0
    assert "i don't have enough information" not in result.response.lower()


def test_how_are_you_does_not_dump_company_knowledge() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    result = orchestrator.handle("p22-2-hi", "Hi, how are you?")
    assert knowledge.queries == []
    assert result.trace.get("should_retrieve") is False
    assert "earkart is" not in result.response.lower()
    assert "digital-first" not in result.response.lower()
    assert recorder.calls[-1]["temperature"] == 0.4


def test_greeting_plus_bluup_still_uses_rag() -> None:
    orchestrator, recorder, *_rest, knowledge = _stack()
    result = orchestrator.handle("p22-2-bluup", "Hi, I want to know about Bluup.")
    assert knowledge.queries
    assert result.trace.get("should_retrieve") is True
    assert recorder.calls[-1]["temperature"] == 0.0


def test_bind_product_for_attribute_questions() -> None:
    assert should_bind_product("what is the price?", "TINY")
    assert bind_product_query("what is the price?", "TINY") == "what is the price of TINY?"
    assert not should_bind_product("What is Earkart?", "TINY")
    state_msg = QueryRewriter()
    from app.services.conversation.models import ConversationState

    state = ConversationState(user_message="what is the warranty?", product="TINY")
    state_msg.apply(state)
    assert state.query_rewritten == "what is the warranty of TINY?"


def test_tell_me_about_bte_during_purchase_uses_rag() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-2-e"
    buy = orchestrator.handle(cid, "I want to buy TINY.")
    assert buy.conversation_goal == ConversationGoal.LEAD
    assert buy.awaiting_field == ""
    assert "phone" not in buy.response.lower()
    asked = orchestrator.handle(cid, "Actually, tell me about BTE first.")
    assert asked.conversation_goal == ConversationGoal.LEAD
    assert asked.product == "TINY"
    assert asked.trace.get("should_retrieve") is True
    assert knowledge.queries
    assert "bte" in knowledge.queries[-1].lower()
    assert lead.leads == []
    assert "phone" not in asked.response.lower()


def test_committed_buyer_can_keep_asking_then_contact() -> None:
    orchestrator, _, lead, *_rest, knowledge = _stack()
    cid = "p22-2-f"
    buy = orchestrator.handle(cid, "I want to buy TINY.")
    assert buy.conversation_goal == ConversationGoal.LEAD
    assert "please provide your phone number" not in buy.response.lower()
    warranty = orchestrator.handle(cid, "What is the warranty?")
    assert warranty.conversation_goal == ConversationGoal.LEAD
    assert any("warranty" in query.lower() for query in knowledge.queries)
    bte = orchestrator.handle(cid, "What about BTE?")
    assert bte.conversation_goal == ConversationGoal.LEAD
    contact = orchestrator.handle(cid, "Okay, I want someone to contact me.")
    assert contact.conversation_goal == ConversationGoal.LEAD
    assert contact.awaiting_field == "name"
    assert lead.leads == []
    assert "what name should we use" not in contact.response.lower()
