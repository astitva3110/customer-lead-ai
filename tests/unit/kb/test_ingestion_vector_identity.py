from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.config import settings
from app.kb.enums import DocumentType, ExtractionMethod
from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord
from app.kb.ingestion.wiring import default_embedding_service
from app.services.knowledge.embedding_service import EmbeddingService


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.identity: Any = None
        self.table_name = ""
        self.schema_ensured = False
        self.fts_ensured = False
        self.upserts: list[dict[str, Any]] = []

    def ensure_schema(self) -> None:
        self.schema_ensured = True

    def ensure_content_fts_index(self) -> None:
        self.fts_ensured = True

    def purge_other_embedding_versions(self) -> int:
        return 0

    def replace_document_vectors(self, document_id: str) -> int:
        return 0

    def document_vectors_exist(self, document_id: str, document_version: int) -> bool:
        return False

    def upsert_chunks(self, **kwargs: Any) -> int:
        self.upserts.append(kwargs)
        return len(kwargs["chunks"])


class _MockEmbeddingProvider:
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def model_revision(self) -> str:
        return "rev"

    @property
    def dimension(self) -> int:
        return 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


def test_default_embedding_service_targets_configured_vector_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = _RecordingVectorStore()
    monkeypatch.setattr(settings, "phase12_vector_table", "chunk_embeddings_identity_test")
    monkeypatch.setattr(
        "app.kb.ingestion.wiring.build_production_vector_store",
        lambda config=None: recording,
    )
    monkeypatch.setattr(
        "app.kb.embedding.factory.create_embedding_provider",
        lambda config: _MockEmbeddingProvider(),
    )

    service = default_embedding_service()
    assert isinstance(service, EmbeddingService)
    assert recording.schema_ensured is True
    assert recording.fts_ensured is True

    record = DocumentRecord(
        document_id="doc-1",
        document_version=1,
        filename="test.pdf",
        mime_type="application/pdf",
        file_hash="file-hash",
        document_type=DocumentType.PRODUCT,
        language="en",
        created_at=datetime.now(timezone.utc),
        storage_path="uploads/test.pdf",
    )
    chunk = Phase12ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_version=1,
        document_type=DocumentType.PRODUCT,
        chunking_algorithm_version=settings.chunking_algorithm_version,
        content="TINY is a hearing aid.",
        embedding_input="TINY is a hearing aid.",
        embedding_input_hash="hash-1",
        token_count=5,
        section_path=["body"],
        page_number=1,
        source_file_hash="file-hash",
        extraction_method=ExtractionMethod.NATIVE,
        created_at=datetime.now(timezone.utc),
        embedding_version=settings.embedding_version,
    )
    result = service.embed_and_index(record=record, chunks=[chunk], title="Test", force=True)

    assert result.inserted == 1
    assert recording.upserts
