"""Phase 12 incremental embedding and PGVector indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.helpers.keyword_query import fts_or_query
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.hashing import content_hash
from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord
from app.kb.vector.phase12_schema import create_phase12_chunk_embedding_table
from app.kb.vector.store import postgres_safe_text

_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Phase12IndexIdentity:
    kb_dataset_version: str
    chunking_algorithm_version: str
    embedding_input_manifest: str
    embedding_version: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int

    @classmethod
    def from_settings(cls) -> Phase12IndexIdentity:
        return cls(
            kb_dataset_version=settings.phase12_kb_dataset_version,
            chunking_algorithm_version=settings.phase12_chunking_algorithm_version,
            embedding_input_manifest=settings.phase12_embedding_input_manifest,
            embedding_version=settings.phase12_embedding_version,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_model_revision=settings.embedding_model_revision,
            embedding_dimension=settings.embedding_dimension,
        )


class Phase12VectorStore:
    """Version-isolated PGVector store — does not touch V1 chunk_embeddings table."""

    def __init__(self, database_url: str | None = None, table_name: str | None = None) -> None:
        self.database_url = database_url or settings.database_url
        self.table_name = table_name or settings.phase12_vector_table
        self.dimension = settings.embedding_dimension
        self.engine = create_engine(self.database_url)
        self._model = create_phase12_chunk_embedding_table(self.table_name, self.dimension)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.identity = Phase12IndexIdentity.from_settings()

    def ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self._model.metadata.create_all(self.engine)

    def exists(self, chunk_id: str) -> bool:
        with self._session_factory() as session:
            stmt = select(self._model.id).where(
                self._model.chunk_id == chunk_id,
                self._model.embedding_version == self.identity.embedding_version,
            )
            return session.execute(stmt).first() is not None

    def document_vectors_exist(self, document_id: str, document_version: int) -> bool:
        with self._session_factory() as session:
            stmt = select(self._model.id).where(
                self._model.document_id == document_id,
                self._model.document_version == document_version,
                self._model.embedding_version == self.identity.embedding_version,
            )
            return session.execute(stmt).first() is not None

    def count(self, *, embedding_version: str | None = None) -> int:
        with self._session_factory() as session:
            stmt = select(self._model.id)
            if embedding_version:
                stmt = stmt.where(self._model.embedding_version == embedding_version)
            return len(session.execute(stmt).all())

    def get_by_chunk_id(self, chunk_id: str) -> dict | None:
        with self._session_factory() as session:
            row = session.execute(
                select(self._model).where(
                    self._model.chunk_id == chunk_id,
                    self._model.embedding_version == self.identity.embedding_version,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._row_to_dict(row)

    def _row_to_dict(self, row) -> dict:
        return {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "document_version": row.document_version,
            "kb_dataset_version": row.kb_dataset_version,
            "chunking_algorithm_version": row.chunking_algorithm_version,
            "embedding_input_manifest": row.embedding_input_manifest,
            "embedding_version": row.embedding_version,
            "embedding_model": row.embedding_model,
            "embedding_model_revision": row.embedding_model_revision,
            "embedding_dimension": row.embedding_dimension,
            "embedding_input_hash": row.embedding_input_hash,
            "source_file_hash": row.source_file_hash,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "section_path": row.section_path,
            "title": row.title,
            "website": row.website,
            "document_type": row.document_type,
            "source_url": row.source_url,
            "canonical_url": row.canonical_url,
            "page_number": row.page_number,
            "source_type": row.source_type,
            "extraction_method": row.extraction_method,
            "split_method": row.split_method,
            "content_hash": row.content_hash,
            "embedded_at": row.embedded_at.isoformat() if row.embedded_at else None,
        }

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        embedding_version: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Read-only cosine similarity search against this Phase 12 table."""
        distance = self._model.embedding.cosine_distance(query_embedding)
        stmt = select(self._model, distance.label("distance")).order_by(distance)
        version = embedding_version or self.identity.embedding_version
        stmt = stmt.where(self._model.embedding_version == version)
        if document_id:
            stmt = stmt.where(self._model.document_id == document_id)
        stmt = stmt.limit(top_k)

        results: list[dict] = []
        with self._session_factory() as session:
            for row, dist in session.execute(stmt).all():
                item = self._row_to_dict(row)
                item["similarity"] = round(1.0 - float(dist), 6)
                item["distance"] = round(float(dist), 6)
                results.append(item)
        return results

    def ensure_content_fts_index(self) -> None:
        """Additive GIN index on chunk content. Does not change embedding columns."""
        if not _SAFE_TABLE_NAME.match(self.table_name):
            raise ValueError(f"Unsafe vector table name: {self.table_name}")
        index_name = f"ix_{self.table_name}_content_fts_simple"
        legacy_name = f"ix_{self.table_name}_content_fts"
        ddl = (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {self.table_name} USING GIN (to_tsvector('simple', content))"
        )
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP INDEX IF EXISTS {legacy_name}"))
            conn.execute(text(ddl))

    def search_keyword(
        self,
        query: str,
        *,
        top_k: int = 10,
        embedding_version: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """PostgreSQL full-text search over existing content. Read-only."""
        cleaned = query.strip()
        if not cleaned or top_k <= 0:
            return []
        tsquery_text = fts_or_query(cleaned)
        if not tsquery_text:
            return []
        tsvector = func.to_tsvector("simple", self._model.content)
        tsquery = func.to_tsquery("simple", tsquery_text)
        rank = func.ts_rank_cd(tsvector, tsquery)
        stmt = select(self._model, rank.label("keyword_score")).order_by(rank.desc())
        version = embedding_version or self.identity.embedding_version
        stmt = stmt.where(self._model.embedding_version == version)
        stmt = stmt.where(tsvector.op("@@")(tsquery))
        if document_id:
            stmt = stmt.where(self._model.document_id == document_id)
        stmt = stmt.limit(top_k)

        results: list[dict] = []
        with self._session_factory() as session:
            for row, score in session.execute(stmt).all():
                item = self._row_to_dict(row)
                item["keyword_score"] = round(float(score or 0.0), 6)
                results.append(item)
        return results

    def upsert_chunks(
        self,
        *,
        record: DocumentRecord,
        chunks: list[Phase12ChunkRecord],
        embeddings: list[list[float]],
        title: str,
        force: bool = False,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        inserted = 0
        with self._session_factory() as session:
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                existing = session.execute(
                    select(self._model).where(
                        self._model.chunk_id == chunk.chunk_id,
                        self._model.embedding_version == self.identity.embedding_version,
                    )
                ).scalar_one_or_none()
                if existing:
                    if force:
                        existing.embedding = embedding
                        existing.embedding_input_hash = chunk.embedding_input_hash
                        existing.content = postgres_safe_text(chunk.content)
                        existing.section_path = chunk.section_path
                        existing.embedded_at = datetime.now(timezone.utc)
                        inserted += 1
                    continue
                row = self._model(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_version=chunk.document_version,
                    kb_dataset_version=self.identity.kb_dataset_version,
                    chunking_algorithm_version=self.identity.chunking_algorithm_version,
                    embedding_input_manifest=self.identity.embedding_input_manifest,
                    embedding_version=self.identity.embedding_version,
                    embedding_provider=self.identity.embedding_provider,
                    embedding_model=self.identity.embedding_model,
                    embedding_model_revision=self.identity.embedding_model_revision,
                    embedding_dimension=self.identity.embedding_dimension,
                    embedding_input_hash=chunk.embedding_input_hash,
                    source_file_hash=chunk.source_file_hash,
                    chunk_index=index,
                    content=postgres_safe_text(chunk.content),
                    section_path=chunk.section_path,
                    title=postgres_safe_text(title),
                    website="uploads",
                    document_type=chunk.document_type.value,
                    source_url=f"file://{record.storage_path}",
                    canonical_url=f"document://{record.document_id}/v{record.document_version}",
                    page_number=chunk.page_number,
                    source_type="pdf",
                    extraction_method=chunk.extraction_method.value,
                    split_method=chunk.split_method,
                    content_hash=content_hash(chunk.content),
                    embedding=embedding,
                    embedded_at=datetime.now(timezone.utc),
                )
                session.add(row)
                inserted += 1
            session.commit()
        return inserted


class Phase12EmbeddingIndexer:
    def __init__(self, store: Phase12VectorStore | None = None) -> None:
        self.store = store or Phase12VectorStore()
        config = EmbeddingConfig.from_settings()
        self.provider = create_embedding_provider(config)

    def embed_and_index(
        self,
        *,
        record: DocumentRecord,
        chunks: list[Phase12ChunkRecord],
        title: str,
        force: bool = False,
    ) -> tuple[int, int]:
        if not force and self.store.document_vectors_exist(record.document_id, record.document_version):
            return 0, 0
        texts = [chunk.embedding_input for chunk in chunks]
        embeddings = self.provider.embed_documents(texts)
        inserted = self.store.upsert_chunks(
            record=record,
            chunks=chunks,
            embeddings=embeddings,
            title=title,
        )
        return len(chunks), inserted
