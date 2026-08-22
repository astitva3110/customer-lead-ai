"""Phase 12 PGVector schema with version isolation from V1 experiment."""

from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.kb.vector.schema import Base

_MODEL_CACHE: dict[str, type] = {}


def create_phase12_chunk_embedding_table(table_name: str, dimension: int):
    cache_key = f"{table_name}:{dimension}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    class Phase12ChunkEmbedding(Base):
        __tablename__ = table_name
        __table_args__ = (
            UniqueConstraint(
                "chunk_id",
                "chunking_algorithm_version",
                "embedding_input_manifest",
                "embedding_model",
                "embedding_model_revision",
                name=f"uq_{table_name}_phase12_identity",
            ),
            {"extend_existing": True},
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        chunk_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
        document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
        document_version: Mapped[int] = mapped_column(Integer, nullable=False)
        kb_dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
        chunking_algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
        embedding_input_manifest: Mapped[str] = mapped_column(String(64), nullable=False)
        embedding_version: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
        embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
        embedding_model: Mapped[str] = mapped_column(String(256), nullable=False)
        embedding_model_revision: Mapped[str] = mapped_column(String(128), nullable=False)
        embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
        embedding_input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
        source_file_hash: Mapped[str] = mapped_column(String(128), nullable=False)
        chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
        content: Mapped[str] = mapped_column(Text, nullable=False)
        section_path: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
        title: Mapped[str] = mapped_column(Text, nullable=False)
        website: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
        document_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
        source_url: Mapped[str] = mapped_column(Text, nullable=False)
        canonical_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
        page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
        source_type: Mapped[str] = mapped_column(String(64), nullable=False)
        extraction_method: Mapped[str] = mapped_column(String(64), nullable=False)
        split_method: Mapped[str] = mapped_column(String(64), nullable=False)
        content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
        embedding: Mapped[list[float]] = mapped_column(Vector(dimension), nullable=False)
        embedded_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
        )

    _MODEL_CACHE[cache_key] = Phase12ChunkEmbedding
    return Phase12ChunkEmbedding
