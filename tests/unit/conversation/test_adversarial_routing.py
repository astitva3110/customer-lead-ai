"""Adversarial routing tests for trap-word messages that confuse regex routing.

These tests document where phrase/regex routing fails and verify that the
semantic router (LLM decision engine) can classify the correct route when given
the right structured output.

Run live LLM checks (costs API tokens):
    set ADVERSARIAL_ROUTING_LIVE_LLM=1
    pytest tests/unit/conversation/test_adversarial_routing.py -m integration -v
"""

from __future__ import annotations

import json
import os

import pytest

from app.config import settings
from app.helpers.conversation_extract import (
    looks_like_acquisition_intent,
    looks_like_device_support_issue,
    looks_like_general_hearing_concern,
)
from app.helpers.semantic_router import (
    SEMANTIC_ROUTER_SYSTEM,
    build_semantic_router_prompt,
    parse_semantic_route,
)
from app.services.conversation.models import ChatMode, ConversationState
from app.services.conversation.query_rewriter import QueryRewriter
from app.services.conversation.router import ChatRouter
from tests.unit.conversation.adversarial_routing_cases import (
    ADVERSARIAL_ROUTING_CASES,
    AdversarialRoutingCase,
)
from tests.unit.conversation.test_semantic_router import ScriptedRouterLLM


def _decision_payload(case: AdversarialRoutingCase) -> str:
    route_map = {
        ChatMode.LEAD: "LEAD",
        ChatMode.KNOWLEDGE: "KNOWLEDGE",
        ChatMode.SUPPORT: "SUPPORT",
    }
    route = route_map.get(case.expected_mode, "GENERAL")
    sales_interest = case.expected_mode == ChatMode.LEAD
    return json.dumps(
        {
            "intent": case.semantic_intent,
            "action": case.semantic_action,
            "needs_rewrite": case.expected_mode == ChatMode.KNOWLEDGE,
            "product": None,
            "confidence": 0.95,
            "route": route,
            "sales_interest": sales_interest,
        }
    )


def _route(message: str, llm: ScriptedRouterLLM | None = None, **state_fields) -> ConversationState:
    state = ConversationState(user_message=message, **state_fields)
    QueryRewriter().normalize(state)
    return ChatRouter(llm=llm).route(state)


def _assert_routed(state: ConversationState, case: AdversarialRoutingCase) -> None:
    assert state.mode == case.expected_mode, (
        f"{case.id}: expected mode {case.expected_mode}, got {state.mode} "
        f"for message={case.message!r}"
    )
    if case.expected_turn_intent:
        assert state.current_turn_intent == case.expected_turn_intent, (
            f"{case.id}: expected turn intent {case.expected_turn_intent}, "
            f"got {state.current_turn_intent}"
        )
    if case.expect_retrieve is not None:
        assert state.trace.get("should_retrieve") is case.expect_retrieve, (
            f"{case.id}: expected should_retrieve={case.expect_retrieve}, "
            f"got {state.trace.get('should_retrieve')}"
        )


@pytest.mark.parametrize("case", ADVERSARIAL_ROUTING_CASES, ids=lambda item: item.id)
def test_adversarial_cases_have_trap_words(case: AdversarialRoutingCase) -> None:
    lowered = case.message.lower()
    assert case.trap_words, f"{case.id} should document trap words"
    assert any(token.lower() in lowered for token in case.trap_words), (
        f"{case.id}: trap words {case.trap_words} not found in message"
    )


@pytest.mark.parametrize("case", ADVERSARIAL_ROUTING_CASES, ids=lambda item: item.id)
def test_adversarial_helper_signals(case: AdversarialRoutingCase) -> None:
    """Document what deterministic helpers think — useful when debugging regex traps."""
    message = case.message
    signals = {
        "acquisition": looks_like_acquisition_intent(message),
        "device_support": looks_like_device_support_issue(message),
        "hearing_concern": looks_like_general_hearing_concern(message),
    }
    assert isinstance(signals["acquisition"], bool)
    assert isinstance(signals["device_support"], bool)
    assert isinstance(signals["hearing_concern"], bool)


@pytest.mark.parametrize("case", ADVERSARIAL_ROUTING_CASES, ids=lambda item: item.id)
def test_phrase_router_documents_known_gaps(case: AdversarialRoutingCase) -> None:
    """Phrase router (no LLM) — known gaps must stay wrong until regex is fixed."""
    routed = _route(case.message)
    if case.phrase_router_known_gap:
        assert routed.mode != case.expected_mode, (
            f"{case.id}: phrase router gap was fixed; update phrase_router_known_gap=False"
        )
        assert routed.mode == case.wrong_regex_guess or routed.mode != case.expected_mode
    else:
        _assert_routed(routed, case)


@pytest.mark.parametrize("case", ADVERSARIAL_ROUTING_CASES, ids=lambda item: item.id)
def test_semantic_router_resolves_adversarial_cases(case: AdversarialRoutingCase) -> None:
    """When the LLM returns the correct structured decision, routing must match."""
    llm = ScriptedRouterLLM(_decision_payload(case))
    routed = _route(case.message, llm)
    if case.expect_semantic_used:
        assert routed.trace.get("semantic_router_used") is True, case.id
    else:
        assert routed.trace.get("semantic_router_used") is not True, case.id
    _assert_routed(routed, case)


@pytest.mark.parametrize(
    "case",
    [item for item in ADVERSARIAL_ROUTING_CASES if item.phrase_router_known_gap],
    ids=lambda item: item.id,
)
def test_semantic_router_fixes_phrase_router_gaps(case: AdversarialRoutingCase) -> None:
    """Cases where regex fails today must be fixed by semantic router."""
    phrase = _route(case.message)
    assert phrase.mode != case.expected_mode

    semantic = _route(case.message, ScriptedRouterLLM(_decision_payload(case)))
    _assert_routed(semantic, case)
    assert semantic.trace.get("semantic_router_used") is True


def test_semantic_fallback_prefers_lead_over_wrong_support() -> None:
    """Acquisition messages must not become support when the LLM misclassifies."""
    wrong_support = json.dumps(
        {
            "intent": "support",
            "action": "start_support",
            "needs_rewrite": False,
            "product": None,
            "confidence": 0.95,
        }
    )
    routed = _route("mujhe hearing aid chaiye", ScriptedRouterLLM(wrong_support))
    assert routed.mode == ChatMode.LEAD
    assert routed.trace.get("semantic_router_used") is not True
    assert routed.trace["semantic_router"]["fallback_reason"] == "prefer_lead"


def test_listening_call_problem_wrong_support_llm_is_known_gap() -> None:
    """Phrase router is correct, but a wrong LLM support label currently misroutes."""
    message = "i have a problem of listening to the call"
    phrase = _route(message)
    assert phrase.mode == ChatMode.KNOWLEDGE

    wrong_support = json.dumps(
        {
            "intent": "support",
            "action": "start_support",
            "needs_rewrite": False,
            "product": None,
            "confidence": 0.95,
        }
    )
    misrouted = _route(message, ScriptedRouterLLM(wrong_support))
    assert misrouted.trace.get("semantic_router_used") is True
    assert misrouted.mode == ChatMode.SUPPORT


def test_hearing_concern_skips_semantic_router_and_stays_knowledge() -> None:
    """General hearing-health messages skip the LLM and must not become support."""
    wrong_support = json.dumps(
        {
            "intent": "support",
            "action": "start_support",
            "needs_rewrite": False,
            "product": None,
            "confidence": 0.95,
        }
    )
    routed = _route(
        "i cannot hear properly on phone calls anymore",
        ScriptedRouterLLM(wrong_support),
    )
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("semantic_router_used") is not True
    assert routed.current_turn_intent == "KNOWLEDGE"


def test_mixed_buy_warranty_stays_knowledge_with_buy_intent() -> None:
    mixed = next(case for case in ADVERSARIAL_ROUTING_CASES if case.id == "mixed_buy_and_warranty")
    routed = _route(mixed.message, ScriptedRouterLLM(_decision_payload(mixed)))
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.lead_intent is True


def _live_llm_configured() -> bool:
    return bool(os.environ.get("ADVERSARIAL_ROUTING_LIVE_LLM", "").strip()) and bool(
        settings.generation_model.strip()
    )


@pytest.mark.integration
@pytest.mark.skipif(not _live_llm_configured(), reason="Set ADVERSARIAL_ROUTING_LIVE_LLM=1 and generation model")
@pytest.mark.parametrize("case", ADVERSARIAL_ROUTING_CASES, ids=lambda item: item.id)
def test_live_semantic_router_classifies_adversarial_cases(case: AdversarialRoutingCase) -> None:
    """Optional live check: real LLM must classify trap messages correctly."""
    from app.providers.llm.factory import get_llm_provider

    llm = get_llm_provider(settings)
    state = ConversationState(user_message=case.message)
    raw = llm.complete(
        SEMANTIC_ROUTER_SYSTEM,
        build_semantic_router_prompt(state),
        temperature=0.0,
        max_tokens=256,
    )
    parsed = parse_semantic_route(raw)
    assert parsed is not None, f"{case.id}: invalid JSON from LLM: {raw[:200]}"
    route_map = {
        ChatMode.LEAD: "LEAD",
        ChatMode.KNOWLEDGE: "KNOWLEDGE",
        ChatMode.SUPPORT: "SUPPORT",
    }
    expected_route = route_map.get(case.expected_mode)
    if case.expected_turn_intent == "GENERAL":
        expected_route = "GENERAL"
    assert parsed.route == expected_route, (
        f"{case.id}: LLM route={parsed.route} intent={parsed.intent} action={parsed.action} "
        f"expected={expected_route} message={case.message!r} why={case.why_tricky}"
    )
