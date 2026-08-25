from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.kb.settings import KbSettings, kb_settings


class RuntimeSettings(BaseSettings):
    """FastAPI runtime settings. Secrets and URLs must come from `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Generation (LiteLLM)
    generation_provider: str = "litellm"
    generation_model: str = ""
    generation_api_base: str = ""
    generation_api_key: str = ""
    generation_temperature: float = 0.0
    generation_conversation_temperature: float = 0.4
    generation_max_tokens: int = 512
    generation_timeout_seconds: float = 60.0
    generation_num_retries: int = 0
    generation_json_mode: bool = True
    generation_extra_body: str = ""
    openai_api_key: str = ""

    # Database — set DATABASE_URL in .env (never commit credentials)
    database_url: str = ""

    # Auth — set JWT_SECRET in .env
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    initial_super_admin_email: str = ""
    initial_super_admin_password: str = ""

    # Chat / lead
    default_country: str = ""

    # Hybrid retrieval (chat path)
    hybrid_retrieval_enabled: bool = True
    vector_candidate_k: int = 20
    keyword_candidate_k: int = 20
    final_retrieval_k: int = 5
    reranker_min_score: float = 0.0
    reranker_provider: str = "lexical_overlap"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_revision: str = ""
    reranker_batch_size: int = 16
    reranker_max_length: int = 512
    reranker_device: str = ""
    embedding_device: str = "cpu"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_model_revision: str = "69da0546e10ee869fbf19f3b1d6c5ac12eb48a16"
    knowledge_vector_top_k: int = 20
    knowledge_keyword_top_k: int = 20
    knowledge_reranker_candidate_k: int = 30
    knowledge_final_context_k: int = 6

    # Diagnostics
    chat_trace_enabled: bool = False
    chat_trace_output_dir: Path = Path("reports/chat_traces")
    chat_trace_retrieval_top_k: int = 10
    chat_trace_include_prompt: bool = False
    chat_trace_include_full_context: bool = False
    chat_trace_text_preview_chars: int = 500
    chat_debug_console: bool = False

    # LangSmith (optional; never hardcode credentials)
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "earkart-chatbot"
    langsmith_endpoint: str = ""


_runtime = RuntimeSettings()


class Settings:
    """Unified view: runtime settings + KB pipeline settings (same `.env`)."""

    __slots__ = ("_runtime", "_kb")

    def __init__(self, runtime: RuntimeSettings, kb: KbSettings) -> None:
        self._runtime = runtime
        self._kb = kb

    def __getattr__(self, name: str):
        if name in RuntimeSettings.model_fields:
            return getattr(self._runtime, name)
        if name in KbSettings.model_fields:
            return getattr(self._kb, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        if name in RuntimeSettings.model_fields:
            setattr(self._runtime, name, value)
            return
        if name in KbSettings.model_fields:
            setattr(self._kb, name, value)
            return
        raise AttributeError(name)


settings = Settings(_runtime, kb_settings)

# Tests construct isolated overrides via RuntimeSettings(...).
Settings = RuntimeSettings
