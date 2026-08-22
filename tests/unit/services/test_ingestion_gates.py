"""Ingestion validation gates — embedding must not run unless both gates pass."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.services.knowledge.embedding_service import EmbeddingService
from app.services.knowledge.ingestion_service import IngestionService
from app.kb.enums import DocumentType, DocumentUploadStatus, ExtractionMethod
from app.kb.ingestion.models import DocumentRecord, ExtractedUnit, ExtractionResult, Phase12ChunkRecord
from app.kb.ingestion.storage import DocumentStorage
from app.kb.ingestion.wiring import build_ingestion_service
from app.kb.models.structured_content import DocumentContent, ParagraphNode
from tests.fixtures.phase12.generate_fixtures import ensure_fixtures


class MockEmbeddingProvider:
    def __init__(self) -> None:
        self.embed_documents_calls: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def model_revision(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        return [[0.1] * 8 for _ in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class MemoryVectorRepository:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.schema_ensured = False

    def ensure_schema(self) -> None:
        self.schema_ensured = True

    def document_vectors_exist(self, document_id: str, document_version: int) -> bool:
        return False

    def upsert_chunks(self, **kwargs: Any) -> int:
        self.upserts.append(kwargs)
        return len(kwargs["chunks"])


class EmptyExtractor:
    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        return ExtractionResult(
            document_id=kwargs["document_id"],
            document_version=kwargs["document_version"],
            title="empty",
            units=[],
            extraction_method=ExtractionMethod.NATIVE,
        )


class GarbageExtractor:
    def extract(self, path: Path, **kwargs: Any) -> ExtractionResult:
        return ExtractionResult(
            document_id=kwargs["document_id"],
            document_version=kwargs["document_version"],
            title="garbage",
            units=[
                ExtractedUnit(
                    document_id=kwargs["document_id"],
                    document_version=kwargs["document_version"],
                    page_number=1,
                    text="@@@ ### $$$ !!! @@@ ### $$$ !!! @@@ ###",
                    extraction_method=ExtractionMethod.NATIVE,
                )
            ],
            extraction_method=ExtractionMethod.NATIVE,
            page_count=1,
        )


class StaticExtractorFactory:
    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    def resolve(self, path: Path) -> Any:
        return self._extractor


class DuplicateChunker:
    def chunk(self, **kwargs: Any) -> tuple[list[Phase12ChunkRecord], list[dict]]:
        record: DocumentRecord = kwargs["record"]
        chunk = _valid_chunk(record, chunk_id="dup")
        clone = chunk.model_copy()
        return [chunk, clone], []


class EmptyChunker:
    def chunk(self, **kwargs: Any) -> tuple[list[Phase12ChunkRecord], list[dict]]:
        return [], []


def _valid_chunk(record: DocumentRecord, *, chunk_id: str) -> Phase12ChunkRecord:
    return Phase12ChunkRecord(
        chunk_id=chunk_id,
        document_id=record.document_id,
        document_version=record.document_version,
        document_type=record.document_type,
        chunking_algorithm_version="phase12.0",
        content="Radius M16 battery life is 270 hours with a size 13 battery.",
        embedding_input="product | Radius M16 battery life is 270 hours with a size 13 battery.",
        embedding_input_hash=f"hash-{chunk_id}",
        token_count=12,
        page_number=1,
        source_file_hash=record.file_hash,
        extraction_method=ExtractionMethod.NATIVE,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "fixtures"
    ensure_fixtures(directory)
    return directory


def _wired_service(
    tmp_path: Path,
    *,
    provider: MockEmbeddingProvider,
    extractor_factory: Any | None = None,
    chunker: Any | None = None,
    vectors: MemoryVectorRepository | None = None,
) -> tuple[IngestionService, MemoryVectorRepository]:
    vectors = vectors or MemoryVectorRepository()
    embedding_service = EmbeddingService(provider, vectors)
    service = build_ingestion_service(
        storage=DocumentStorage(base_dir=tmp_path / "documents"),
        embedding_service=embedding_service,
    )
    if extractor_factory is not None:
        service.extractor_factory = extractor_factory
    if chunker is not None:
        service.chunker = chunker
    return service, vectors


def test_valid_pdf_calls_embedding_after_gates(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, vectors = _wired_service(tmp_path, provider=provider)
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        dry_run=False,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.INDEXED
    assert provider.embed_documents_calls
    assert vectors.upserts
    artifact_root = tmp_path / "documents" / result.document.document_id
    assert (artifact_root / "extracted" / "pages.json").exists()
    assert (artifact_root / "cleaned" / "document.json").exists()
    assert (artifact_root / "chunks" / "chunks.json").exists()


def test_invalid_pdf_does_not_embed(tmp_path: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(tmp_path, provider=provider)
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a pdf")
    try:
        result = service.ingest_file(corrupt, document_type=DocumentType.OTHER, embed=True)
        assert result.document.status == DocumentUploadStatus.FAILED
    except Exception:
        pass
    assert provider.embed_documents_calls == []


def test_empty_extraction_does_not_embed(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(
        tmp_path,
        provider=provider,
        extractor_factory=StaticExtractorFactory(EmptyExtractor()),
    )
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.FAILED
    assert "empty" in (result.document.error or "")
    assert provider.embed_documents_calls == []


def test_document_quality_failure_does_not_embed(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(
        tmp_path,
        provider=provider,
        extractor_factory=StaticExtractorFactory(GarbageExtractor()),
    )
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.FAILED
    assert "Document validation failed" in (result.document.error or "")
    assert provider.embed_documents_calls == []


def test_successful_document_validation_reaches_chunking(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(tmp_path, provider=provider)
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        dry_run=True,
        embed=False,
    )
    assert result.document.status == DocumentUploadStatus.CHUNKED
    assert result.chunks
    assert provider.embed_documents_calls == []


def test_chunk_validation_failure_does_not_embed(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(
        tmp_path,
        provider=provider,
        chunker=EmptyChunker(),
    )
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.FAILED
    assert "Chunk validation failed" in (result.document.error or "")
    assert provider.embed_documents_calls == []


def test_duplicate_chunk_ids_do_not_embed(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(
        tmp_path,
        provider=provider,
        chunker=DuplicateChunker(),
    )
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.FAILED
    assert "duplicate_chunk" in (result.document.error or "")
    assert provider.embed_documents_calls == []


def test_failed_ingestion_persists_failed_status(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(
        tmp_path,
        provider=provider,
        extractor_factory=StaticExtractorFactory(EmptyExtractor()),
    )
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    stored = service.get_document_status(result.document.document_id)
    assert stored is not None
    assert stored.status == DocumentUploadStatus.FAILED
    failed_path = service.documents.failed_dir / f"{result.document.document_id}.json"
    assert failed_path.exists()


class ExistingVectorRepository(MemoryVectorRepository):
    def document_vectors_exist(self, document_id: str, document_version: int) -> bool:
        return True


class RaisingVectorRepository(MemoryVectorRepository):
    def upsert_chunks(self, **kwargs: Any) -> int:
        raise RuntimeError("upsert failed before commit")


def test_existing_vectors_mark_indexed_without_reembedding(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, vectors = _wired_service(
        tmp_path,
        provider=provider,
        vectors=ExistingVectorRepository(),
    )
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.INDEXED
    assert result.indexed
    assert provider.embed_documents_calls == []
    assert vectors.upserts == []


def test_vector_upsert_failure_marks_failed(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(
        tmp_path,
        provider=provider,
        vectors=RaisingVectorRepository(),
    )
    with pytest.raises(RuntimeError, match="upsert failed before commit"):
        service.ingest_file(
            fixture_dir / "digital_product.pdf",
            document_type=DocumentType.PRODUCT,
            embed=True,
        )
    stored = next(iter(service.documents.list_documents()))
    assert stored.status == DocumentUploadStatus.FAILED
    assert provider.embed_documents_calls


def test_post_commit_artifact_failure_keeps_indexed(tmp_path: Path, fixture_dir: Path) -> None:
    provider = MockEmbeddingProvider()
    service, vectors = _wired_service(tmp_path, provider=provider)
    original = service._write_processed_artifact

    def maybe_fail(record, result):
        if result.indexed:
            raise OSError("disk full")
        original(record, result)

    service._write_processed_artifact = maybe_fail  # type: ignore[method-assign]
    result = service.ingest_file(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
        embed=True,
    )
    assert result.document.status == DocumentUploadStatus.INDEXED
    assert result.indexed
    assert vectors.upserts
    finalization = tmp_path / "documents" / result.document.document_id / "reports" / "finalization.json"
    assert finalization.exists()


def test_background_wrapper_marks_failed_on_unexpected_error(tmp_path: Path, fixture_dir: Path) -> None:
    from app.routers.knowledge import run_background_ingestion

    provider = MockEmbeddingProvider()
    service, _vectors = _wired_service(tmp_path, provider=provider)
    record, _ = service.upload_document(
        fixture_dir / "digital_product.pdf",
        document_type=DocumentType.PRODUCT,
    )
    service.mark_processing(record.document_id)

    def boom(document_id: str, **kwargs: Any):
        raise RuntimeError("unexpected crash")

    service.process_document = boom  # type: ignore[method-assign]
    run_background_ingestion(service, record.document_id)
    stored = service.get_document_status(record.document_id)
    assert stored is not None
    assert stored.status == DocumentUploadStatus.FAILED
