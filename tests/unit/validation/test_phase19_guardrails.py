from __future__ import annotations

from app.services.conversation.guardrail import guardrail_check
from app.services.conversation.models import ChatMode
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


def test_prompt_injection_variants() -> None:
    assert guardrail_check("ignore all previous") == "prompt_injection"
    assert guardrail_check("developer mode") == "prompt_injection"
    assert guardrail_check("jailbreak") == "prompt_injection"


def test_abuse_blocked() -> None:
    assert guardrail_check("kill yourself") == "unsupported"


def test_irrelevant_weather_is_not_blocked_by_fast_guardrail() -> None:
    assert guardrail_check("What's the weather in New York?") is None


def test_why_buy_routes_to_knowledge() -> None:
    orchestrator, recorder, *_ = make_graph_stack(knowledge=FakeKnowledge())
    result = orchestrator.handle("buy-q", "Why should I buy from Earkart?")
    assert result.mode == ChatMode.KNOWLEDGE
    assert recorder.call_count == 1
