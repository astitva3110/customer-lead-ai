from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.litellm_provider import LiteLLMProvider
from app.kb.evaluation.models import RankedHit
from tests.validation.harness import RecordingLiteLLM, make_litellm_generation


def _hit() -> RankedHit:
    return RankedHit(
        rank=1,
        similarity=0.9,
        chunk_id="c1",
        document_id="d1",
        section_path=["About"],
        token_count=8,
        content_type="paragraph",
        text="Earkart sells hearing aids.",
        document_title="About",
    )


def test_factory_uses_litellm() -> None:
    provider = get_llm_provider(Settings(generation_provider="litellm", generation_model="openai/Qwen/Qwen2.5-7B-Instruct"))
    assert isinstance(provider, LiteLLMProvider)


def test_successful_model_routing() -> None:
    generation, recorder, _ = make_litellm_generation()
    result = generation.generate("What is Earkart?", [_hit()])
    assert recorder.call_count == 1
    assert recorder.calls[0]["model"] == "openai/Qwen/Qwen2.5-7B-Instruct"
    assert result.grounded is True


@pytest.mark.parametrize(
    "behavior",
    ["timeout", "provider_error", "rate_limit", "auth", "invalid_model"],
)
def test_provider_errors_are_normalized(behavior: str) -> None:
    generation, recorder, _ = make_litellm_generation(RecordingLiteLLM(behavior))
    result = generation.generate("What is Earkart?", [_hit()])
    assert recorder.call_count == 1
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE


def test_malformed_response_fails_closed() -> None:
    generation, recorder, _ = make_litellm_generation(RecordingLiteLLM("malformed"))
    result = generation.generate("What is Earkart?", [_hit()])
    assert recorder.call_count == 1
    assert result.grounded is False


def test_api_key_not_logged(caplog, monkeypatch) -> None:
    recorder = RecordingLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=recorder.completion))
    settings = Settings(
        generation_model="openai/Qwen/Qwen2.5-7B-Instruct",
        generation_api_key="super-secret-key-value",
        generation_json_mode=True,
    )
    provider = LiteLLMProvider(settings)
    with caplog.at_level(logging.INFO):
        provider.complete("sys", "user", temperature=0.0, max_tokens=32)
    assert "super-secret-key-value" not in caplog.text
    assert recorder.calls[0]["api_key"] == "super-secret-key-value"
