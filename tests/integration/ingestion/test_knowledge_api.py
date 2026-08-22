from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.knowledge.embedding_service import EmbeddingService
from app.kb.enums import DocumentType, DocumentUploadStatus
from app.kb.ingestion.storage import DocumentStorage
from app.kb.ingestion.wiring import build_ingestion_service
from app.main import app
from app.dependencies import get_ingestion_service
from app.domain.entities import UserRole
from tests.fixtures.phase12.generate_fixtures import ensure_fixtures
from tests.unit.auth.helpers import as_user, make_user
from tests.unit.services.test_ingestion_gates import MemoryVectorRepository, MockEmbeddingProvider


def test_upload_and_status_endpoints(tmp_path) -> None:
    fixture_dir = tmp_path / "fixtures"
    ensure_fixtures(fixture_dir)
    provider = MockEmbeddingProvider()
    storage = DocumentStorage(base_dir=tmp_path / "documents")
    service = build_ingestion_service(
        storage=storage,
        embedding_service=EmbeddingService(provider, MemoryVectorRepository()),
    )

    app.dependency_overrides[get_ingestion_service] = lambda: service
    admin, _password = make_user(role=UserRole.ADMIN)
    try:
        with as_user(admin):
            client = TestClient(app)
            with (fixture_dir / "digital_product.pdf").open("rb") as handle:
                response = client.post(
                    "/api/v1/knowledge/documents",
                    files={"file": ("digital_product.pdf", handle, "application/pdf")},
                    data={"document_type": DocumentType.PRODUCT.value},
                )
            assert response.status_code == 202
            body = response.json()
            assert body["status"] == "PROCESSING"
            document_id = body["document_id"]

            status = client.get(f"/api/v1/knowledge/documents/{document_id}")
        assert status.status_code == 200
        payload = status.json()
        assert payload["document_id"] == document_id
        assert payload["status"] in {
            DocumentUploadStatus.INDEXED.value.upper(),
            DocumentUploadStatus.EMBEDDED.value.upper(),
            DocumentUploadStatus.FAILED.value.upper(),
        }
        assert "progress" in payload
        if payload["status"] == "INDEXED":
            assert provider.embed_documents_calls
    finally:
        app.dependency_overrides.clear()


def test_invalid_upload_returns_400(tmp_path) -> None:
    storage = DocumentStorage(base_dir=tmp_path / "documents")
    service = build_ingestion_service(
        storage=storage,
        embedding_service=EmbeddingService(MockEmbeddingProvider(), MemoryVectorRepository()),
    )
    app.dependency_overrides[get_ingestion_service] = lambda: service
    admin, _password = make_user(role=UserRole.ADMIN)
    try:
        with as_user(admin):
            client = TestClient(app)
            response = client.post(
                "/api/v1/knowledge/documents",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()
