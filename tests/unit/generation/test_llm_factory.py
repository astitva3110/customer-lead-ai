from app.config import Settings
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.litellm_provider import LiteLLMProvider


def test_factory_always_returns_litellm() -> None:
    settings = Settings(generation_provider="litellm", generation_model="openai/Qwen/Qwen3-1.7B")
    provider = get_llm_provider(settings)
    assert isinstance(provider, LiteLLMProvider)


def test_openai_config_still_uses_litellm() -> None:
    settings = Settings(
        generation_provider="openai",
        generation_model="openai/gpt-4o-mini",
        openai_api_key="sk-test",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, LiteLLMProvider)
    assert provider._settings.generation_model == "openai/gpt-4o-mini"
