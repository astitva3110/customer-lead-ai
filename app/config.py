from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""

    # Generation — LiteLLM is the only gateway. Switch backends via these fields.
    generation_provider: str = "litellm"
    generation_model: str = "openai/Qwen/Qwen2.5-7B-Instruct"
    generation_api_base: str = ""
    generation_api_key: str = ""
    generation_temperature: float = 0.0
    generation_conversation_temperature: float = 0.4
    generation_max_tokens: int = 512
    generation_timeout_seconds: float = 60.0
    generation_num_retries: int = 0
    generation_json_mode: bool = True
    generation_extra_body: str = ""
    embedding_provider: str = "qwen_local"
    retriever: str = "keyword"
    crawl_limit: int = 500
    crawl_max_depth: int = 10

    knowledge_dir: Path = Path("data/knowledge")
    raw_dir: Path = Path("data/raw")
    cleaned_dir: Path = Path("data/cleaned")
    canonical_dir: Path = Path("data/canonical")
    retrieval_dir: Path = Path("data/retrieval")
    chunks_dir: Path = Path("data/chunks")
    reports_dir: Path = Path("reports")

    chunk_max_tokens: int = 512
    chunk_emergency_overlap_tokens: int = 32
    chunk_chars_per_token: float = 4.0

    # Phase 11 — embedding + vector store
    kb_dataset_version: str = "2026-08-17-v1"
    chunking_algorithm_version: str = "phase10.6"
    embedding_input_manifest: str = "2026-08-17-v1"
    embedding_version: str = "qwen_Qwen3-Embedding-0.6B_v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_model_revision: str = "69da0546e10ee869fbf19f3b1d6c5ac12eb48a16"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 8
    embedding_normalize: bool = True
    embedding_device: str = "cpu"
    embedding_query_instruction: str = (
        "Given a user question about Earkart products, policies, or investor information, "
        "retrieve relevant passages that answer the question"
    )
    database_url: str = "postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot"
    vector_table: str = "chunk_embeddings"
    vector_smoke_table: str = "chunk_embeddings_smoke"
    vector_benchmark_table: str = "chunk_embeddings_benchmark"

    # Phase 12 — document-first ingestion (separate from V1 web-scrape experiment)
    phase12_kb_dataset_version: str = "phase12-v1"
    phase12_chunking_algorithm_version: str = "phase12.0"
    phase12_embedding_input_manifest: str = "phase12-v1"
    phase12_embedding_version: str = "qwen_Qwen3-Embedding-0.6B_phase12_v1"
    phase12_vector_table: str = "chunk_embeddings_phase12"
    phase14_vector_smoke_table: str = "chunk_embeddings_kb_v2_smoke"
    phase14_smoke_chunk_count: int = 10
    documents_dir: Path = Path("data/documents")
    max_upload_bytes: int = 52_428_800  # 50 MB

    # Phase 16 — KB V3 chunking experiment (no embed / no PGVector writes)
    phase16_kb_dataset_version: str = "phase16-v1"
    phase16_chunking_algorithm_version: str = "phase16.0"
    phase16_embedding_input_manifest: str = "phase16-v1"
    phase16_target_min_tokens: int = 30
    phase16_target_max_tokens: int = 150

    # Hybrid Retrieval V1 — candidate generation + rerank + threshold
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

    # Knowledge chat path — starting values only, not claimed optimal
    knowledge_vector_top_k: int = 20
    knowledge_keyword_top_k: int = 20
    knowledge_reranker_candidate_k: int = 30
    knowledge_final_context_k: int = 6

    # Chat diagnostics — observe the existing pipeline. Default off.
    chat_trace_enabled: bool = False
    chat_trace_output_dir: Path = Path("reports/chat_traces")
    chat_trace_retrieval_top_k: int = 10
    chat_trace_include_prompt: bool = False
    chat_trace_include_full_context: bool = False
    chat_trace_text_preview_chars: int = 500
    chat_debug_console: bool = False

    # Authentication
    jwt_secret: str = "dev-only-change-me-use-32-chars-min"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    initial_super_admin_email: str = ""
    initial_super_admin_password: str = ""


settings = Settings()
