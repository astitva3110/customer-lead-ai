from app.config import Settings
from app.interfaces.providers.llm import LLMProvider
from app.providers.llm.litellm_provider import LiteLLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Always LiteLLM. Switch OpenAI / local OpenAI-compatible models via Settings only."""
    return LiteLLMProvider(settings, profile="generation")


def get_semantic_router_provider(settings: Settings) -> LLMProvider:
    """Routing LLM — can use OpenRouter while generation stays on local Ollama."""
    return LiteLLMProvider(settings, profile="semantic_router")


def semantic_router_model_label(settings: Settings) -> str:
    return (settings.semantic_router_model or settings.generation_model or "").strip()
