from __future__ import annotations

import logging
import time
from typing import Any

from app.services.conversation.models import ConversationState
from app.services.generation.models import GenerationResult, ungrounded_fallback
from app.interfaces.providers.llm import LLMProvider
from app.helpers.conversation_prompt import (
    CONVERSATION_SYSTEM_PROMPT,
    build_conversation_user_prompt,
    parse_conversation_answer,
)
from app.helpers.conversation_reply import conversational_fallback, repeats_recent_assistant
from app.helpers.conversation_turn import has_greeting_prefix, last_assistant_text
from app.helpers.generation_json import extract_json_object
from app.helpers.generation_prompt import GENERATION_SYSTEM_PROMPT, build_generation_user_prompt
from app.helpers.generation_validate import explain_generation_payload, validate_generation_payload
from app.services.diagnostics.recorder import current_trace, record_error, record_generation, record_grounding

logger = logging.getLogger(__name__)

_CREATED_CLAIM = (
    "will contact you shortly",
    "get in touch with you shortly",
    "get back to you shortly",
    "details have been shared",
    "created a support ticket",
    "ticket has been created",
    "lead has been created",
    "i've created",
    "i have created",
)


class GenerationService:
    """Grounded RAG generation plus compact conversational replies."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        temperature: float = 0.0,
        conversation_temperature: float = 0.4,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._conversation_temperature = conversation_temperature
        self._max_tokens = max_tokens

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def conversation_temperature(self) -> float:
        return self._conversation_temperature

    def generate(self, query: str, hits: list[Any]) -> GenerationResult:
        if not hits:
            if current_trace():
                record_grounding(
                    {
                        "grounded_requested": True,
                        "grounded_returned": None,
                        "validator_result": False,
                        "validator_reason": "empty_hits",
                        "returned_source_ids": [],
                        "valid_source_ids": [],
                        "invalid_source_ids": [],
                        "available_source_ids": [],
                        "source_id_validation_passed": False,
                        "fallback_triggered": True,
                    }
                )
            return ungrounded_fallback()
        if not self._llm.is_configured:
            logger.warning("generation skipped: LLM is not configured")
            return ungrounded_fallback()
        started = time.perf_counter()
        raw, user_prompt = self._complete_once(query, hits)
        latency_ms = (time.perf_counter() - started) * 1000
        if raw is None:
            if current_trace():
                record_generation(
                    provider="litellm",
                    model="",
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    latency_ms=latency_ms,
                    llm_call_count=1,
                    prompt=user_prompt,
                    system_prompt=GENERATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    raw_output=None,
                    error=RuntimeError("generation LLM call failed"),
                )
            return ungrounded_fallback()
        allowed = {hit.chunk_id for hit in hits}
        payload = extract_json_object(raw)
        result = validate_generation_payload(payload, allowed)
        explained = explain_generation_payload(payload, allowed)
        if current_trace():
            record_grounding(explained)
            parsed = None if result is None else {
                "grounded": result.grounded,
                "answer": result.answer,
                "source_ids": result.source_ids,
            }
            record_generation(
                provider="litellm",
                model="",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                latency_ms=latency_ms,
                llm_call_count=1,
                result=parsed,
                prompt=user_prompt,
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_output=raw,
                parsed_payload=payload,
                explained=explained,
            )
        if result is None:
            logger.warning("generation failed closed: invalid or ungrounded JSON")
            return ungrounded_fallback()
        return result

    def converse(self, state: ConversationState) -> ConversationState:
        if (state.response or "").strip():
            return state
        fallback = conversational_fallback(state)
        if not self._llm.is_configured:
            state.response = fallback
            return state
        started = time.perf_counter()
        user_prompt = build_conversation_user_prompt(state)
        try:
            raw = self._llm.complete(
                CONVERSATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=self._conversation_temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.exception("conversation LLM call failed")
            if current_trace():
                record_error("generation", exc, recoverable=True, fallback_used=True)
            state.response = fallback
            return state
        state.trace = dict(state.trace or {})
        state.trace["generation_ms"] = round((time.perf_counter() - started) * 1000, 3)
        state.trace["generation_temperature"] = self._conversation_temperature
        state.trace["llm_call_count"] = int(state.trace.get("llm_call_count") or 0) + 1
        answer = parse_conversation_answer(raw, state)
        if repeats_recent_assistant(state, answer):
            answer = fallback
        if _claims_success(answer) and not state.trace.get("tool_called"):
            answer = fallback
        if current_trace():
            record_generation(
                provider="litellm",
                model="",
                temperature=self._conversation_temperature,
                max_tokens=self._max_tokens,
                latency_ms=float(state.trace["generation_ms"]),
                llm_call_count=int(state.trace["llm_call_count"]),
                result={"grounded": None, "answer": answer, "source_ids": []},
                system_prompt=CONVERSATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_output=raw,
            )
        state.response = answer
        return state

    def _complete_once(self, query: str, hits: list[Any]) -> tuple[str | None, str]:
        user_prompt = build_generation_user_prompt(
            query,
            hits,
            acknowledge_greeting=has_greeting_prefix(query),
        )
        try:
            raw = self._llm.complete(
                GENERATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception:
            logger.exception("generation LLM call failed")
            return None, user_prompt
        return raw, user_prompt


def _claims_success(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(token in lowered for token in _CREATED_CLAIM)
