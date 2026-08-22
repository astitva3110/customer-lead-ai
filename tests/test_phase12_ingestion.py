"""Phase 12 document-first ingestion tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.enums import DocumentType, ExtractionMethod
from app.kb.ingestion.embedding_input import build_embedding_input, embedding_input_hash
from app.kb.ingestion.extraction.base import get_extractor
from app.kb.ingestion.indexing import Phase12IndexIdentity
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.quality import ChunkQualityGate, validate_chunk
from app.kb.ingestion.service import DocumentIngestionService
from app.kb.ingestion.storage import DocumentStorage
from app.kb.ingestion.structure import build_structured_document
from app.kb.ingestion.validation import sha256_file, validate_upload_file
from tests.fixtures.phase12.generate_fixtures import ensure_fixtures

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase12"


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("phase12_fixtures")
    ensure_fixtures(directory)
    return directory


@pytest.fixture
def isolated_storage(tmp_path) -> DocumentStorage:
    return DocumentStorage(base_dir=tmp_path / "documents")


@pytest.fixture
def service(isolated_storage) -> DocumentIngestionService:
    return DocumentIngestionService(storage=isolated_storage)


class TestValidation:
    def test_pdf_validation(self, fixture_dir):
        result = validate_upload_file(fixture_dir / "digital_product.pdf")
        assert result.ok
        assert result.extension == ".pdf"

    def test_docx_validation(self, fixture_dir):
        result = validate_upload_file(fixture_dir / "policy_returns.docx")
        assert result.ok
        assert result.extension == ".docx"

    def test_rejects_unsupported_extension(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        result = validate_upload_file(path)
        assert not result.ok

    def test_sha256_stability(self, fixture_dir):
        path = fixture_dir / "digital_product.pdf"
        assert sha256_file(path) == sha256_file(path)


class TestStorage:
    def test_duplicate_file_detection(self, service, fixture_dir):
        path = fixture_dir / "digital_product.pdf"
        first, _ = service.upload_document(path, document_type=DocumentType.PRODUCT)
        second, dup = service.upload_document(path, document_type=DocumentType.PRODUCT)
        assert first.document_id
        assert dup
        assert second.document_id == first.document_id


class TestExtraction:
    def test_native_pdf_extraction(self, fixture_dir):
        extractor = get_extractor(fixture_dir / "digital_product.pdf")
        result = extractor.extract(
            fixture_dir / "digital_product.pdf",
            document_id="doc-1",
            document_version=1,
            allow_ocr=False,
        )
        assert result.units

    def test_docx_extraction(self, fixture_dir):
        extractor = get_extractor(fixture_dir / "policy_returns.docx")
        result = extractor.extract(
            fixture_dir / "policy_returns.docx",
            document_id="doc-2",
            document_version=1,
        )
        assert any("Return window" in unit.text for unit in result.units)

    def test_table_preservation(self, fixture_dir):
        extractor = get_extractor(fixture_dir / "product_specs_table.pdf")
        result = extractor.extract(
            fixture_dir / "product_specs_table.pdf",
            document_id="doc-3",
            document_version=1,
            allow_ocr=False,
        )
        combined = "\n".join(unit.text for unit in result.units)
        assert "Battery Life" in combined
        assert "270 Hrs" in combined

    def test_page_provenance(self, fixture_dir):
        extractor = get_extractor(fixture_dir / "digital_product.pdf")
        result = extractor.extract(
            fixture_dir / "digital_product.pdf",
            document_id="doc-4",
            document_version=1,
            allow_ocr=False,
        )
        assert any(unit.page_number == 1 for unit in result.units)


class TestStructureAndChunking:
    def test_section_hierarchy(self, fixture_dir):
        extractor = get_extractor(fixture_dir / "radius_m16.docx")
        extraction = extractor.extract(
            fixture_dir / "radius_m16.docx",
            document_id="doc-5",
            document_version=1,
        )
        structured, _ = build_structured_document(extraction)
        assert structured.title

    def test_semantic_chunk_boundaries(self, service, fixture_dir):
        result = service.ingest_file(
            fixture_dir / "radius_m16.docx",
            document_type=DocumentType.PRODUCT,
            dry_run=True,
            embed=False,
        )
        assert result.chunks
        assert all(chunk.token_count <= 512 for chunk in result.chunks)

    def test_deterministic_chunk_ids(self, fixture_dir):
        service_a = DocumentIngestionService(storage=DocumentStorage(base_dir=fixture_dir / "store_a"))
        service_b = DocumentIngestionService(storage=DocumentStorage(base_dir=fixture_dir / "store_b"))
        r1 = service_a.ingest_file(fixture_dir / "digital_product.pdf", document_type=DocumentType.PRODUCT, dry_run=True)
        r2 = service_b.ingest_file(fixture_dir / "digital_product.pdf", document_type=DocumentType.PRODUCT, dry_run=True)
        assert [c.chunk_id for c in r1.chunks] == [c.chunk_id for c in r2.chunks]

    def test_deterministic_embedding_input(self):
        first = build_embedding_input(
            document_type=DocumentType.PRODUCT,
            title="Radius M16",
            section_path=["Technical Specifications"],
            content="Battery Life: 270 Hrs",
        )
        second = build_embedding_input(
            document_type=DocumentType.PRODUCT,
            title="Radius M16",
            section_path=["Technical Specifications"],
            content="Battery Life: 270 Hrs",
        )
        assert embedding_input_hash(first) == embedding_input_hash(second)


class TestQualityGate:
    def test_token_ceiling(self, service, fixture_dir):
        result = service.ingest_file(
            fixture_dir / "digital_product.pdf",
            document_type=DocumentType.PRODUCT,
            dry_run=True,
        )
        report = ChunkQualityGate().evaluate(result.chunks)
        assert report["statistics"]["max_token_count"] <= 512

    def test_ocr_debris_suppression_flag(self):
        garbage = Phase12ChunkRecord(
            chunk_id="x",
            document_id="d",
            document_version=1,
            document_type=DocumentType.OTHER,
            chunking_algorithm_version="phase12.0",
            content="@@@ ### $$$ !!! @@@ ### $$$ !!!",
            embedding_input="test",
            embedding_input_hash="abc",
            token_count=10,
            source_file_hash="hash",
            extraction_method=ExtractionMethod.NATIVE,
            created_at=datetime.now(timezone.utc),
        )
        assert "ocr_garbage" in validate_chunk(garbage)


class TestVersioning:
    def test_pgvector_version_isolation(self):
        identity = Phase12IndexIdentity.from_settings()
        assert identity.embedding_version != "qwen_Qwen3-Embedding-0.6B_v1"
        assert identity.kb_dataset_version == "phase12-v1"

    def test_dry_run_never_writes_vectors(self, service, fixture_dir, monkeypatch):
        called = {"upsert": False}

        def fail_upsert(*args, **kwargs):
            called["upsert"] = True
            raise AssertionError("dry run must not upsert vectors")

        monkeypatch.setattr(
            "app.kb.ingestion.indexing.Phase12EmbeddingIndexer.embed_and_index",
            fail_upsert,
        )
        result = service.ingest_file(
            fixture_dir / "digital_product.pdf",
            document_type=DocumentType.PRODUCT,
            dry_run=True,
            embed=False,
        )
        assert result.dry_run
        assert not called["upsert"]
