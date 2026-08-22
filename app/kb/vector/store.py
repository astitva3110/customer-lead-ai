"""PGVector storage layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.embedding.versioning import VectorIndexIdentity
from app.kb.vector.schema import create_chunk_embedding_table


def postgres_safe_text(value: str) -> str:
    """PostgreSQL TEXT cannot store NUL bytes; frozen chunk files remain unchanged."""
    return value.replace("\x00", "")


@dataclass
class VectorRecordInput:
    chunk_id: str
    document_id: str
    document_version: int
    kb_dataset_version: str
    chunking_algorithm_version: str
    embedding_input_manifest: str
    embedding_version: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    chunk_index: int
    content: str
    section_path: list[str]
    title: str
    website: str
    document_type: str
    source_url: str
    canonical_url: str
    page_number: int | None
    source_type: str
    extraction_method: str
    split_method: str
    content_hash: str
    embedding: list[float]

    @classmethod
    def from_chunk(
        cls,
        *,
        chunk: ProductionChunkRecord,
        embedding: list[float],
        identity: VectorIndexIdentity,
        provider_name: str,
    ) -> VectorRecordInput:
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_version=chunk.document_version,
            kb_dataset_version=identity.kb_dataset_version,
            chunking_algorithm_version=identity.chunking_algorithm_version,
            embedding_input_manifest=identity.embedding_input_manifest,
            embedding_version=identity.embedding_version,
            embedding_provider=provider_name,
            embedding_model=identity.embedding_model,
            embedding_model_revision=identity.embedding_model_revision,
            embedding_dimension=identity.embedding_dimension,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            section_path=list(chunk.section_path),
            title=chunk.title,
            website=chunk.website,
            document_type=chunk.document_type.value,
            source_url=chunk.source_url,
            canonical_url=chunk.canonical_url,
            page_number=chunk.page_number,
            source_type=chunk.source_type.value,
            extraction_method=chunk.extraction_method.value,
            split_method=chunk.split_method.value,
            content_hash=chunk.provenance.content_hash,
            embedding=embedding,
        )


class VectorStore:
    def __init__(
        self,
        database_url: str,
        table_name: str,
        dimension: int,
        *,
        engine: Engine | None = None,
    ) -> None:
        self.database_url = database_url
        self.table_name = table_name
        self.dimension = dimension
        self.engine = engine or create_engine(database_url)
        self._model = create_chunk_embedding_table(table_name, dimension)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self._model.metadata.create_all(self.engine)

    def exists(self, chunk_id: str, identity: VectorIndexIdentity) -> bool:
        with self._session_factory() as session:
            stmt = select(self._model.id).where(
                self._model.chunk_id == chunk_id,
                self._model.chunking_algorithm_version == identity.chunking_algorithm_version,
                self._model.embedding_input_manifest == identity.embedding_input_manifest,
                self._model.embedding_model == identity.embedding_model,
                self._model.embedding_model_revision == identity.embedding_model_revision,
            )
            return session.execute(stmt).first() is not None

    def upsert_many(self, records: list[VectorRecordInput], *, force: bool = False) -> int:
        if not records:
            return 0
        inserted = 0
        with self._session_factory() as session:
            for record in records:
                existing = session.execute(
                    select(self._model).where(
                        self._model.chunk_id == record.chunk_id,
                        self._model.chunking_algorithm_version == record.chunking_algorithm_version,
                        self._model.embedding_input_manifest == record.embedding_input_manifest,
                        self._model.embedding_model == record.embedding_model,
                        self._model.embedding_model_revision == record.embedding_model_revision,
                    )
                ).scalar_one_or_none()
                if existing and not force:
                    continue
                if existing and force:
                    session.delete(existing)
                    session.flush()
                row = self._model(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    document_version=record.document_version,
                    kb_dataset_version=record.kb_dataset_version,
                    chunking_algorithm_version=record.chunking_algorithm_version,
                    embedding_input_manifest=record.embedding_input_manifest,
                    embedding_version=record.embedding_version,
                    embedding_provider=record.embedding_provider,
                    embedding_model=record.embedding_model,
                    embedding_model_revision=record.embedding_model_revision,
                    embedding_dimension=record.embedding_dimension,
                    chunk_index=record.chunk_index,
                    content=postgres_safe_text(record.content),
                    section_path=record.section_path,
                    title=postgres_safe_text(record.title),
                    website=record.website,
                    document_type=record.document_type,
                    source_url=record.source_url,
                    canonical_url=record.canonical_url,
                    page_number=record.page_number,
                    source_type=record.source_type,
                    extraction_method=record.extraction_method,
                    split_method=record.split_method,
                    content_hash=record.content_hash,
                    embedding=record.embedding,
                    embedded_at=datetime.now(timezone.utc),
                )
                session.add(row)
                inserted += 1
            session.commit()
        return inserted

    def count(self, *, embedding_version: str | None = None) -> int:
        with self._session_factory() as session:
            stmt = select(self._model.id)
            if embedding_version:
                stmt = stmt.where(self._model.embedding_version == embedding_version)
            return len(session.execute(stmt).all())

    def get_by_chunk_id(self, chunk_id: str) -> dict | None:
        with self._session_factory() as session:
            row = session.execute(
                select(self._model).where(self._model.chunk_id == chunk_id)
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
            "embedding_provider": row.embedding_provider,
            "embedding_model": row.embedding_model,
            "embedding_model_revision": row.embedding_model_revision,
            "embedding_dimension": row.embedding_dimension,
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
        website: str | None = None,
        document_type: str | None = None,
        source_type: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        distance = self._model.embedding.cosine_distance(query_embedding)
        stmt = select(self._model, distance.label("distance")).order_by(distance)
        if embedding_version:
            stmt = stmt.where(self._model.embedding_version == embedding_version)
        if website:
            stmt = stmt.where(self._model.website == website)
        if document_type:
            stmt = stmt.where(self._model.document_type == document_type)
        if source_type:
            stmt = stmt.where(self._model.source_type == source_type)
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
