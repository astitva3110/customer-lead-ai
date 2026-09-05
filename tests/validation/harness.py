from __future__ import annotations

import json
import statistics
from types import SimpleNamespace
from typing import Any

from app.config import Settings
from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.repositories.memory_lead import InMemoryLeadAdapter
from app.repositories.memory_ticket import InMemoryTicketAdapter
from app.repositories.conversation import InMemoryConversationRepository
from app.graph.factory import build_conversation_orchestrator
from app.providers.llm.litellm_provider import LiteLLMProvider

from app.helpers.conversation_turn import is_greeting_only, is_short_no, is_short_yes
from app.helpers.semantic_router import SEMANTIC_ROUTER_MARKER


def conversation_stub_answer(user: str) -> str:
    try:
        payload = json.loads(user)
    except json.JSONDecodeError:
        payload = {}
    message = str(payload.get("current_message") or user).strip()
    last = str(payload.get("last_assistant") or "")
    product = str(payload.get("product") or "that") or "that"
    situation = str(payload.get("situation") or "")
    lowered = message.lower()
    if is_greeting_only(message):
        return "Hey! 👋 How can I help you today?"
    if "just said they want to buy" in situation.lower():
        return (
            f"Absolutely! {product} is a great choice. "
            "I can help with any questions, and I can connect you with our sales team whenever you're ready."
        )
    if is_short_yes(message):
        return f"Sure — I can cover {product} whenever you're ready."
    if is_short_no(message) or ("phone" in lowered and "not" in lowered):
        return "That's fine. We can continue without that."
    if "issue" in situation.lower() or "not working" in lowered or "isn't working" in lowered:
        return "I'm sorry you're dealing with that. Tell me a little about what's happening, and I'll see how I can help."
    if "shared personal information" in situation.lower() and payload.get("name_known"):
        name = str(payload.get("name") or "").strip()
        if name:
            return f"Nice to meet you, {name}!"
        return "Nice to meet you. I've noted that."
    if "want to buy" in lowered or "interested" in lowered or "considering a purchase" in situation.lower():
        return f"Got it — I can keep helping with {product}."
    if last:
        return f"Understood. I can keep helping with {product}."
    return f"Got it — I can help you with {product}."


def _parse_prompt_sources(user: str) -> list[tuple[str, str]]:
    parts = user.split("[SOURCE_ID: ")
    parsed: list[tuple[str, str]] = []
    for part in parts[1:]:
        source_id, rest = part.split("]", 1)
        body = rest.split("Content:\n", 1)[-1]
        body = body.split("User question:", 1)[0]
        parsed.append((source_id.strip(), body.strip()))
    return parsed


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def latency_summary(values: list[float]) -> dict[str, float | None | int]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": percentile(values, 95),
        "max": round(max(values), 3),
    }


class RecordingLiteLLM:
    """LiteLLM completion stand-in. Exercises LiteLLMProvider, not a second LLM path."""

    def __init__(self, behavior: str = "grounded") -> None:
        self.behavior = behavior
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def router_call_count(self) -> int:
        return sum(1 for call in self.calls if _is_semantic_router_call(call))

    @property
    def generation_call_count(self) -> int:
        return self.call_count - self.router_call_count

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.behavior == "timeout":
            raise TimeoutError("LiteLLM timeout")
        if self.behavior == "provider_error":
            raise RuntimeError("LiteLLM provider error")
        if self.behavior == "rate_limit":
            raise RuntimeError("RateLimitError: 429")
        if self.behavior == "auth":
            raise RuntimeError("AuthenticationError: invalid API key")
        if self.behavior == "invalid_model":
            raise RuntimeError("NotFoundError: model not found")
        if self.behavior == "malformed":
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))])
        if _is_semantic_router_call(kwargs):
            payload = {
                "canonical_query": "",
                "route": "KNOWLEDGE",
                "product": None,
                "sales_interest": False,
                "diverge": False,
                "sub_questions": [],
                "confidence": 0.0,
            }
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=16, total_tokens=24),
            )
        user = kwargs["messages"][1]["content"]
        if "[SOURCE_ID:" not in user:
            payload = {"answer": conversation_stub_answer(user)}
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )
        question = user.split("User question:\n", 1)[-1].split("\n", 1)[0].strip()
        parsed = _parse_prompt_sources(user)
        source_ids = [item[0] for item in parsed]
        contents = [item[1] for item in parsed]
        lowered_q = question.lower()
        lowered_ctx = " ".join(contents).lower()
        supported = bool(source_ids)
        if "signia" in lowered_q:
            supported = False
        if "in hours" in lowered_q:
            supported = False
        if not supported:
            payload = {
                "grounded": False,
                "answer": INSUFFICIENT_INFORMATION_MESSAGE,
                "source_ids": [],
            }
        else:
            snippet = (contents[0] if contents else "Retrieved knowledge.").strip().split("\n")[0][:240]
            payload = {"grounded": True, "answer": snippet, "source_ids": [source_ids[0]]}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )


def _is_semantic_router_call(kwargs: dict[str, Any]) -> bool:
    messages = kwargs.get("messages") or []
    if not messages:
        return False
    first = messages[0]
    content = first.get("content") if isinstance(first, dict) else getattr(first, "content", "")
    return SEMANTIC_ROUTER_MARKER in str(content or "")


def install_litellm(monkeypatch, recorder: RecordingLiteLLM):
    monkeypatch.setitem(__import__("sys").modules, "litellm", recorder)


def make_litellm_generation(recorder: RecordingLiteLLM | None = None) -> tuple[GenerationService, RecordingLiteLLM, LiteLLMProvider]:
    recorder = recorder or RecordingLiteLLM()
    import sys
    from types import ModuleType

    module = ModuleType("litellm")
    module.completion = recorder.completion
    sys.modules["litellm"] = module
    settings = Settings(
        generation_provider="litellm",
        generation_model="openai/Qwen/Qwen2.5-7B-Instruct",
        generation_json_mode=True,
        generation_api_key="test-key",
    )
    provider = LiteLLMProvider(settings)
    return (
        GenerationService(
            provider,
            temperature=0.0,
            conversation_temperature=0.4,
            max_tokens=512,
        ),
        recorder,
        provider,
    )


def make_graph_stack(*, knowledge, recorder: RecordingLiteLLM | None = None, lead_tool=None, ticket_tool=None):
    generation, recorder, provider = make_litellm_generation(recorder)
    lead_tool = lead_tool or InMemoryLeadAdapter()
    ticket_tool = ticket_tool or InMemoryTicketAdapter()
    store = InMemoryConversationRepository()
    orchestrator = build_conversation_orchestrator(
        knowledge=knowledge,
        generation=generation,
        lead_tool=lead_tool,
        ticket_tool=ticket_tool,
        store=store,
        model_used=provider._settings.generation_model,
    )
    return orchestrator, recorder, lead_tool, ticket_tool, store, provider
