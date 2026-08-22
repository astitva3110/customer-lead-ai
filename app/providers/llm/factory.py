from app.config import Settings
from app.interfaces.providers.llm import LLMProvider
from app.providers.llm.litellm_provider import LiteLLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Always LiteLLM. Switch OpenAI / local OpenAI-compatible models via Settings only."""
    return LiteLLMProvider(settings)
