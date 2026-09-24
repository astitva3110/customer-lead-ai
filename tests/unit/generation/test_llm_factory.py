from app.config import Settings
from app.providers.llm.factory import get_llm_provider, get_semantic_router_provider, semantic_router_model_label
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


def test_semantic_router_provider_uses_separate_openrouter_settings() -> None:
    settings = Settings(
        generation_model="ollama/qwen3:4b",
        generation_api_base="http://127.0.0.1:11434",
        generation_api_key="ollama",
        semantic_router_model="openrouter/qwen/qwen-2.5-7b-instruct",
        semantic_router_api_base="https://openrouter.ai/api/v1",
        semantic_router_api_key="sk-or-test",
    )
    generation = get_llm_provider(settings)
    router = get_semantic_router_provider(settings)
    assert generation.profile == "generation"
    assert router.profile == "semantic_router"
    assert generation._resolved_model() == "ollama/qwen3:4b"
    assert router._resolved_model() == "openrouter/qwen/qwen-2.5-7b-instruct"
    assert router._resolved_api_base() == "https://openrouter.ai/api/v1"
    assert router._resolved_api_key() == "sk-or-test"
    assert semantic_router_model_label(settings) == "openrouter/qwen/qwen-2.5-7b-instruct"
