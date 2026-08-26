from __future__ import annotations

import json

from app.helpers.semantic_router import (
    SEMANTIC_ROUTER_MARKER,
    parse_semantic_route,
    recovery_user_prompt,
)
from app.helpers.state_manager import conversation_status_label, current_status_label
from app.services.conversation.models import (
    ChatMode,
    ConversationGoal,
    ConversationState,
    LeadStage,
    LeadStatus,
)
from app.services.conversation.query_rewriter import QueryRewriter
from app.services.conversation.router import ChatRouter


def _payload(**overrides) -> str:
    data = {
        "route": "KNOWLEDGE",
        "product": None,
        "sales_interest": False,
        "diverge": False,
        "sub_questions": [],
        "confidence": 0.95,
    }
    data.update(overrides)
    return json.dumps(data)


class ScriptedRouterLLM:
    def __init__(self, outputs: str | list[str], *, configured: bool = True) -> None:
        self.outputs = [outputs] if isinstance(outputs, str) else list(outputs)
        self.call_count = 0
        self.is_configured = configured
        self.last_system = ""
        self.last_user = ""
        self.last_temperature: float | None = None

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        del max_tokens
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        self.last_temperature = temperature
        if self.outputs:
            return self.outputs.pop(0)
        return "{}"


def _route(message: str, llm: ScriptedRouterLLM | None = None, **state_fields) -> ConversationState:
    state = ConversationState(user_message=message, **state_fields)
    QueryRewriter().normalize(state)
    return ChatRouter(llm=llm).route(state)


def test_parse_semantic_route_schema() -> None:
    parsed = parse_semantic_route(
        _payload(route="KNOWLEDGE", product="TINY", sales_interest=True, diverge=False, sub_questions=["What is TINY's warranty?"])
    )
    assert parsed is not None
    assert parsed.route == "KNOWLEDGE"
    assert parsed.product == "TINY"
    assert parsed.sales_interest is True
    assert parsed.sub_questions == ["What is TINY's warranty?"]


def test_parse_semantic_route_rejects_invalid() -> None:
    assert parse_semantic_route("not json") is None
    assert parse_semantic_route('{"route": "SUPPORT"}') is None
    assert parse_semantic_route('{"answer": "hi"}') is None


def test_recovery_prompt_is_compact() -> None:
    prompt = recovery_user_prompt('{"normalized_query":"What is TINY?"}', "blah " * 80)
    assert "valid JSON" in prompt
    assert len(prompt) < 500


def test_buy_tiny_routes_to_lead() -> None:
    routed = _route("I need to buy TINY", ScriptedRouterLLM(_payload(route="LEAD", product="TINY", sales_interest=True)))
    assert routed.mode == ChatMode.LEAD
    assert routed.product == "TINY"
    assert routed.sales_interest is True
    assert routed.trace["semantic_router_used"] is True
    assert routed.trace["semantic_router"]["route"] == "LEAD"


def test_buy_tinny_uses_rewriter_then_lead() -> None:
    state = ConversationState(user_message="I need to buy tinny")
    rewritten = QueryRewriter().normalize(state)
    assert "TINY" in rewritten.rewritten_query
    routed = ChatRouter(llm=ScriptedRouterLLM(_payload(route="LEAD", product="TINY", sales_interest=True))).route(state)
    assert routed.mode == ChatMode.LEAD
    assert routed.product == "TINY"
    assert routed.trace["semantic_router"]["normalized_query"]


def test_buy_plus_features_is_knowledge_with_sales_interest() -> None:
    routed = _route(
        "I want TINY. What are its features?",
        ScriptedRouterLLM(
            _payload(
                route="KNOWLEDGE",
                product="TINY",
                sales_interest=True,
                sub_questions=["What are the features of TINY?"],
            )
        ),
    )
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.sales_interest is True
    assert conversation_status_label(routed) == "SALE"
    assert current_status_label(routed) == "KNOWLEDGE"
    assert routed.product == "TINY"


def test_warranty_spelling_is_knowledge() -> None:
    routed = _route(
        "what is tinny warrenty?",
        ScriptedRouterLLM(_payload(route="KNOWLEDGE", product="TINY", sales_interest=False, sub_questions=["What is TINY's warranty?"])),
    )
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.product == "TINY"
    assert routed.sales_interest is False


def test_pronoun_warranty_keeps_active_product() -> None:
    routed = _route(
        "What is its warranty?",
        ScriptedRouterLLM(_payload(route="KNOWLEDGE", product="TINY", sales_interest=True)),
        product="TINY",
        conversation_goal=ConversationGoal.SALES,
        lead_intent=True,
        sales_interest=True,
        conversation_history=[
            {"role": "user", "content": "I want to buy TINY"},
            {"role": "assistant", "content": "TINY is a compact hearing aid."},
        ],
    )
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.product == "TINY"
    assert routed.sales_interest is True
    assert conversation_status_label(routed) == "SALE"
    assert current_status_label(routed) == "KNOWLEDGE"


def test_how_are_you_is_general() -> None:
    routed = _route("how are you?", ScriptedRouterLLM(_payload(route="GENERAL", product=None, sales_interest=False)))
    assert routed.current_turn_intent == "GENERAL"
    assert routed.trace.get("should_retrieve") is False
    assert routed.product == ""


def test_thanks_is_general() -> None:
    routed = _route("thanks", ScriptedRouterLLM(_payload(route="GENERAL", product=None, sales_interest=False)))
    assert routed.current_turn_intent == "GENERAL"
    assert routed.trace.get("should_retrieve") is False


def test_person_question_is_knowledge() -> None:
    routed = _route("who is Rohit Misra?", ScriptedRouterLLM(_payload(route="KNOWLEDGE", product=None, sales_interest=False)))
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.product == ""


def test_divergent_feature_and_warranty() -> None:
    routed = _route(
        "What are TINY's features and what is its warranty?",
        ScriptedRouterLLM(
            _payload(
                route="KNOWLEDGE",
                product="TINY",
                sales_interest=False,
                diverge=True,
                sub_questions=["What are TINY's features?", "What is TINY's warranty?"],
            )
        ),
    )
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("diverge") is True
    assert routed.trace.get("sub_questions") == ["What are TINY's features?", "What is TINY's warranty?"]
    assert routed.sales_interest is False


def test_lead_then_general_does_not_reset_sale() -> None:
    routed = _route(
        "thanks",
        ScriptedRouterLLM(_payload(route="GENERAL", product="TINY", sales_interest=True)),
        product="TINY",
        conversation_goal=ConversationGoal.SALES,
        lead_intent=True,
        sales_interest=True,
        lead_status=LeadStatus.COLLECTING,
        lead_stage=LeadStage.IN_PROGRESS,
        conversation_history=[
            {"role": "user", "content": "I want to buy TINY"},
            {"role": "assistant", "content": "TINY is a great choice."},
        ],
    )
    assert current_status_label(routed) == "GENERAL"
    assert conversation_status_label(routed) == "SALE"
    assert routed.product == "TINY"
    assert routed.sales_interest is True


def test_invalid_json_recovers_then_uses_second_payload() -> None:
    llm = ScriptedRouterLLM(["not-json", _payload(route="LEAD", product="TINY", sales_interest=True)])
    routed = _route("I need to buy TINY", llm)
    assert llm.call_count == 2
    assert routed.mode == ChatMode.LEAD
    assert routed.trace["semantic_router"]["recovery_attempted"] is True
    assert routed.trace["semantic_router_used"] is True


def test_invalid_json_after_recovery_falls_back_to_phrase_router() -> None:
    llm = ScriptedRouterLLM(["not-json", "still-not-json"])
    routed = _route("I need to buy TINY", llm)
    assert llm.call_count == 2
    assert routed.mode == ChatMode.LEAD
    assert routed.trace["semantic_router_used"] is False
    assert routed.trace["semantic_router"]["fallback_reason"] == "invalid_json"
    assert routed.trace["phrase_router"]["turn_intent"] == "LEAD_INTENT"


def test_low_confidence_does_not_route_to_general() -> None:
    llm = ScriptedRouterLLM(_payload(route="GENERAL", product=None, sales_interest=False, confidence=0.2))
    routed = _route("who is Rohit Misra?", llm)
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.trace["semantic_router_used"] is False
    assert routed.trace["semantic_router"]["fallback_reason"] == "low_confidence"


def test_buy_plus_warranty_does_not_stay_on_lead() -> None:
    llm = ScriptedRouterLLM(_payload(route="LEAD", product="TINY", sales_interest=True, confidence=0.99))
    routed = _route("I need to buy TINY. What is its warranty?", llm)
    assert routed.mode == ChatMode.KNOWLEDGE
    assert routed.trace.get("should_retrieve") is True
    assert routed.sales_interest is True
    assert conversation_status_label(routed) == "SALE"


def test_phrase_router_still_used_without_llm() -> None:
    routed = _route("I want to buy TINY")
    assert routed.mode == ChatMode.LEAD
    assert routed.trace.get("semantic_router_used") is not True
    assert routed.trace.get("phrase_router")


def test_semantic_prompt_is_compact_and_not_full_history() -> None:
    history = [{"role": "user", "content": f"turn {i} " + ("x" * 80)} for i in range(12)]
    history.append({"role": "assistant", "content": "ok " + ("y" * 80)})
    llm = ScriptedRouterLLM(_payload(route="GENERAL"))
    _route("thanks", llm, conversation_history=history, product="TINY")
    assert SEMANTIC_ROUTER_MARKER in llm.last_system
    assert "turn 0" not in llm.last_user
    assert "answer" not in llm.last_system.lower() or "Do not answer the user" in llm.last_system
    assert len(llm.last_user) < 1200


def test_compare_phrase_and_semantic_on_clean_buy() -> None:
    message = "I need to buy TINY"
    phrase = _route(message)
    semantic = _route(message, ScriptedRouterLLM(_payload(route="LEAD", product="TINY", sales_interest=True)))
    assert phrase.mode == semantic.mode == ChatMode.LEAD
    assert phrase.product == semantic.product == "TINY"
    assert semantic.trace["semantic_router_used"] is True
    assert phrase.trace.get("semantic_router_used") is not True
