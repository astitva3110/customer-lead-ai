from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class KbSettings(BaseSettings):
    """KB ingestion, embedding, and offline pipeline settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedding_provider: str = "qwen_local"
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
    kb_dataset_version: str = "2026-08-17-v1"
    chunking_algorithm_version: str = "phase10.6"
    embedding_input_manifest: str = "2026-08-17-v1"
    embedding_version: str = "qwen_Qwen3-Embedding-0.6B_v1"
    vector_table: str = "chunk_embeddings"
    vector_smoke_table: str = "chunk_embeddings_smoke"
    vector_benchmark_table: str = "chunk_embeddings_benchmark"

    phase12_kb_dataset_version: str = "phase12-v1"
    phase12_chunking_algorithm_version: str = "phase12.0"
    phase12_embedding_input_manifest: str = "phase12-v1"
    phase12_embedding_version: str = "qwen_Qwen3-Embedding-0.6B_phase12_v1"
    phase12_vector_table: str = "chunk_embeddings_phase12"
    phase14_vector_smoke_table: str = "chunk_embeddings_kb_v2_smoke"
    phase14_smoke_chunk_count: int = 10
    phase16_kb_dataset_version: str = "phase16-v1"
    phase16_chunking_algorithm_version: str = "phase16.0"
    phase16_embedding_input_manifest: str = "phase16-v1"
    phase16_target_min_tokens: int = 30
    phase16_target_max_tokens: int = 150

    chunk_max_tokens: int = 512
    chunk_emergency_overlap_tokens: int = 32
    chunk_chars_per_token: float = 4.0
    max_upload_bytes: int = 52_428_800

    knowledge_dir: Path = Path("data/knowledge")
    raw_dir: Path = Path("data/raw")
    cleaned_dir: Path = Path("data/cleaned")
    canonical_dir: Path = Path("data/canonical")
    retrieval_dir: Path = Path("data/retrieval")
    chunks_dir: Path = Path("data/chunks")
    reports_dir: Path = Path("reports")
    documents_dir: Path = Path("data/documents")


kb_settings = KbSettings()
