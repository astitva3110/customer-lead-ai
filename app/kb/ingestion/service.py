"""Backward-compatible facade over IngestionService."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.services.knowledge.embedding_service import EmbeddingService
from app.services.knowledge.ingestion_service import IngestionService
from app.kb.chunking.config import ChunkingConfig
from app.kb.enums import DocumentType
from app.kb.ingestion.models import DocumentRecord, IngestionResult
from app.kb.ingestion.quality import ChunkQualityGate
from app.kb.ingestion.storage import DocumentStorage
from app.kb.ingestion.wiring import build_ingestion_service


class DocumentIngestionService:
    """API-ready service for PDF/DOC/DOCX upload → chunk → embed → index."""

    def __init__(
        self,
        storage: DocumentStorage | None = None,
        quality_gate: ChunkQualityGate | None = None,
        chunking_config: ChunkingConfig | None = None,
        embedding_service: EmbeddingService | None = None,
        embedding_factory: Callable[[], EmbeddingService] | None = None,
    ) -> None:
        self.storage = storage or DocumentStorage()
        self._inner = build_ingestion_service(
            storage=self.storage,
            quality_gate=quality_gate,
            chunking_config=chunking_config,
            embedding_service=embedding_service,
            embedding_factory=embedding_factory,
        )

    @property
    def ingestion(self) -> IngestionService:
        return self._inner

    def upload_document(
        self,
        file_path: Path,
        *,
        document_type: DocumentType,
        language: str = "en",
    ) -> tuple[DocumentRecord, bool]:
        return self._inner.upload_document(file_path, document_type=document_type, language=language)

    def process_document(
        self,
        document_id: str,
        *,
        allow_ocr: bool = True,
        dry_run: bool = False,
        embed: bool = True,
    ) -> IngestionResult:
        return self._inner.process_document(
            document_id,
            allow_ocr=allow_ocr,
            dry_run=dry_run,
            embed=embed,
        )

    def chunk_document(self, document_id: str, *, allow_ocr: bool = True) -> IngestionResult:
        return self._inner.chunk_document(document_id, allow_ocr=allow_ocr)

    def embed_document(self, document_id: str) -> IngestionResult:
        return self._inner.embed_document(document_id)

    def index_document(self, document_id: str) -> IngestionResult:
        return self._inner.index_document(document_id)

    def get_document_status(self, document_id: str) -> DocumentRecord | None:
        return self._inner.get_document_status(document_id)

    def ingest_file(
        self,
        file_path: Path,
        *,
        document_type: DocumentType,
        language: str = "en",
        allow_ocr: bool = True,
        dry_run: bool = False,
        embed: bool = True,
    ) -> IngestionResult:
        return self._inner.ingest_file(
            file_path,
            document_type=document_type,
            language=language,
            allow_ocr=allow_ocr,
            dry_run=dry_run,
            embed=embed,
        )
