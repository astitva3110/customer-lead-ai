"""Document ingestion orchestrator — no PDF/SDK/pgvector imports."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.interfaces.providers.chunker import DocumentChunker
from app.interfaces.providers.cleaner import DocumentCleaner
from app.interfaces.providers.pdf import ExtractorFactory
from app.interfaces.repositories.document_repository import DocumentRepository
from app.services.knowledge.chunk_validation import validate_chunk_set
from app.services.knowledge.document_validation import validate_extracted_document
from app.services.knowledge.embedding_service import EmbeddingService
from app.kb.enums import DocumentType, DocumentUploadStatus
from app.kb.ingestion.models import DocumentRecord, IngestionResult
from app.kb.ingestion.quality import ChunkQualityGate
from app.kb.ingestion.validation import validate_upload_file

logger = logging.getLogger(__name__)

_BLOCKING_CHUNK_ISSUES = frozenset(
    {
        "no_valid_chunks",
        "duplicate_chunk_ids",
        "duplicate_chunks",
        "missing_document_id",
        "invalid_embedding_input",
    }
)


class IngestionService:
    """Coordinates validate → store → extract → clean → chunk → embed → index."""

    def __init__(
        self,
        *,
        documents: DocumentRepository,
        extractor_factory: ExtractorFactory,
        cleaner: DocumentCleaner,
        chunker: DocumentChunker,
        quality_gate: ChunkQualityGate | None = None,
        embedding_service: EmbeddingService | None = None,
        embedding_factory: Callable[[], EmbeddingService] | None = None,
    ) -> None:
        self.documents = documents
        self.extractor_factory = extractor_factory
        self.cleaner = cleaner
        self.chunker = chunker
        self.quality_gate = quality_gate or ChunkQualityGate()
        self._embedding_service = embedding_service
        self._embedding_factory = embedding_factory

    def upload_document(
        self,
        file_path: Path,
        *,
        document_type: DocumentType,
        language: str = "en",
    ) -> tuple[DocumentRecord, bool]:
        record, validation, is_duplicate = self.documents.store_upload(
            file_path,
            document_type=document_type,
            language=language,
        )
        if not validation.ok:
            return record, False
        if is_duplicate:
            return record, True
        return record, False

    def mark_processing(self, document_id: str) -> None:
        self.documents.mark_status(document_id, DocumentUploadStatus.PROCESSING)

    def get_document_status(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

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
        record, is_duplicate = self.upload_document(
            file_path,
            document_type=document_type,
            language=language,
        )
        if is_duplicate:
            return IngestionResult(document=record, duplicate=True, dry_run=dry_run)
        if not record.document_id:
            raise ValueError(record.error or "Upload failed")
        return self.process_document(
            record.document_id,
            allow_ocr=allow_ocr,
            dry_run=dry_run,
            embed=embed and not dry_run,
        )

    def process_document(
        self,
        document_id: str,
        *,
        allow_ocr: bool = True,
        dry_run: bool = False,
        embed: bool = True,
    ) -> IngestionResult:
        record = self.documents.get(document_id)
        if record is None:
            raise ValueError(f"Unknown document_id: {document_id}")
        if record.status == DocumentUploadStatus.DUPLICATE:
            return IngestionResult(document=record, duplicate=True, dry_run=dry_run)

        path = Path(record.storage_path)
        validation = validate_upload_file(path)
        if not validation.ok:
            return self._fail(document_id, validation.error or "Validation failed", dry_run=dry_run)

        self.documents.mark_status(document_id, DocumentUploadStatus.EXTRACTING)
        try:
            extractor = self.extractor_factory.resolve(path)
            extraction = extractor.extract(
                path,
                document_id=record.document_id,
                document_version=record.document_version,
                title=record.title,
                allow_ocr=allow_ocr,
            )
            self.documents.mark_status(
                document_id,
                DocumentUploadStatus.EXTRACTED,
                pages=extraction.page_count or len({unit.page_number for unit in extraction.units if unit.page_number}),
            )
            self.documents.write_json(
                document_id,
                "extracted/pages.json",
                extraction.model_dump(mode="json"),
            )
            self.documents.write_json(
                document_id,
                "reports/extraction.json",
                {
                    "extraction_method": extraction.extraction_method.value,
                    "page_count": extraction.page_count,
                    "unit_count": len(extraction.units),
                    "ocr_used": extraction.ocr_used,
                },
            )

            self.documents.mark_status(document_id, DocumentUploadStatus.CLEANING)
            structured, clean_stats = self.cleaner.clean(extraction)
            self.documents.mark_status(document_id, DocumentUploadStatus.CLEANED)
            self.documents.write_json(
                document_id,
                "cleaned/document.json",
                structured.model_dump(mode="json"),
            )
            self.documents.write_json(
                document_id,
                "reports/cleaning.json",
                {"stats": _jsonable(clean_stats)},
            )

            self.documents.mark_status(document_id, DocumentUploadStatus.VALIDATING)
            document_issues = validate_extracted_document(extraction, record)
            if document_issues:
                return self._fail(
                    document_id,
                    "Document validation failed: " + ", ".join(document_issues),
                    dry_run=dry_run,
                    extra={"document_issues": document_issues},
                )
            self.documents.mark_status(document_id, DocumentUploadStatus.VALIDATED)

            self.documents.mark_status(document_id, DocumentUploadStatus.CHUNKING)
            chunks, suppressed = self.chunker.chunk(
                record=record,
                extraction=extraction,
                structured=structured,
            )
            quality = self.quality_gate.evaluate(chunks)
            passed_chunks = quality["passed"]
            chunk_issues = validate_chunk_set(chunks, passed_chunks)
            blocking_chunk_issues = [issue for issue in chunk_issues if issue in _BLOCKING_CHUNK_ISSUES]
            self.documents.write_json(
                document_id,
                "chunks/chunks.json",
                {"chunks": [chunk.model_dump(mode="json") for chunk in passed_chunks]},
            )
            self.documents.write_json(
                document_id,
                "reports/chunking.json",
                {
                    "statistics": quality.get("statistics", {}),
                    "failed": quality.get("failed", []),
                    "set_issues": chunk_issues,
                    "suppressed_count": len(suppressed),
                },
            )
            if blocking_chunk_issues:
                return self._fail(
                    document_id,
                    "Chunk validation failed: " + ", ".join(blocking_chunk_issues),
                    dry_run=dry_run,
                    extra={"chunk_issues": chunk_issues},
                    chunks=len(chunks),
                )

            self.documents.mark_status(
                document_id,
                DocumentUploadStatus.CHUNKED,
                chunks=len(passed_chunks),
            )
            record = self.documents.get(document_id) or record
            result = IngestionResult(
                document=record,
                extraction=extraction,
                chunks=passed_chunks,
                suppressed_chunks=suppressed + quality["failed"],
                quality=quality,
                dry_run=dry_run,
            )
            self._write_processed_artifact(record, result)

            if dry_run or not embed:
                return result

            embedding_service = self._require_embedding_service()
            self.documents.mark_status(document_id, DocumentUploadStatus.EMBEDDING)
            vectors_committed = False
            try:
                outcome = embedding_service.embed_and_index(
                    record=record,
                    chunks=passed_chunks,
                    title=extraction.title,
                )
                vectors_committed = outcome.committed
                if outcome.already_indexed or outcome.committed:
                    self._mark_indexed(
                        document_id,
                        embedded=len(passed_chunks),
                        chunks=len(passed_chunks),
                    )
                    result.embedded = True
                    result.indexed = True
                elif outcome.embedded_count:
                    self.documents.mark_status(
                        document_id,
                        DocumentUploadStatus.EMBEDDED,
                        embedded=outcome.embedded_count,
                    )
                    result.embedded = True
                record = self.documents.get(document_id) or record
                result.document = record
                try:
                    self._write_processed_artifact(record, result)
                except Exception:
                    logger.exception(
                        "artifact finalization failed after vector commit for document_id=%s",
                        document_id,
                    )
                    self._report_finalization_failure(
                        document_id,
                        "processed artifact write failed after vector commit",
                    )
                return result
            except Exception:
                if vectors_committed:
                    logger.exception(
                        "post-commit ingestion error for document_id=%s; keeping INDEXED",
                        document_id,
                    )
                    self._mark_indexed(
                        document_id,
                        embedded=len(passed_chunks),
                        chunks=len(passed_chunks),
                    )
                    self._report_finalization_failure(document_id, "post-commit ingestion error")
                    record = self.documents.get(document_id) or record
                    result.document = record
                    result.embedded = True
                    result.indexed = True
                    return result
                raise
        except Exception as exc:
            self._fail(document_id, str(exc), dry_run=dry_run)
            raise

    def chunk_document(self, document_id: str, *, allow_ocr: bool = True) -> IngestionResult:
        return self.process_document(document_id, allow_ocr=allow_ocr, dry_run=True, embed=False)

    def embed_document(self, document_id: str) -> IngestionResult:
        return self.process_document(document_id, dry_run=False, embed=True)

    def index_document(self, document_id: str) -> IngestionResult:
        return self.embed_document(document_id)

    def record_unexpected_failure(self, document_id: str, error: str | None = None) -> None:
        record = self.documents.get(document_id)
        if record is None:
            return
        if record.status in {
            DocumentUploadStatus.INDEXED,
            DocumentUploadStatus.FAILED,
            DocumentUploadStatus.DUPLICATE,
        }:
            return
        self._fail(document_id, error or "Unexpected ingestion failure", dry_run=False)

    def _mark_indexed(self, document_id: str, *, embedded: int, chunks: int) -> None:
        self.documents.mark_status(
            document_id,
            DocumentUploadStatus.INDEXED,
            embedded=embedded,
            chunks=chunks,
        )

    def _report_finalization_failure(self, document_id: str, error: str) -> None:
        try:
            self.documents.write_json(
                document_id,
                "reports/finalization.json",
                {
                    "document_id": document_id,
                    "index_status": DocumentUploadStatus.INDEXED.value,
                    "artifact_error": error,
                },
            )
        except Exception:
            logger.exception("could not write finalization report for document_id=%s", document_id)

    def _require_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is not None:
            return self._embedding_service
        if self._embedding_factory is None:
            raise RuntimeError("Embedding service is not configured")
        self._embedding_service = self._embedding_factory()
        return self._embedding_service

    def _fail(
        self,
        document_id: str,
        error: str,
        *,
        dry_run: bool,
        extra: dict[str, Any] | None = None,
        chunks: int | None = None,
    ) -> IngestionResult:
        self.documents.mark_status(
            document_id,
            DocumentUploadStatus.FAILED,
            error=error,
            chunks=chunks,
        )
        payload: dict[str, Any] = {"document_id": document_id, "error": error}
        if extra:
            payload.update(extra)
        failed_path = self.documents.failed_dir / f"{document_id}.json"
        failed_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        record = self.documents.get(document_id)
        if record is None:
            raise ValueError(error)
        return IngestionResult(document=record, dry_run=dry_run)

    def _write_processed_artifact(self, record: DocumentRecord, result: IngestionResult) -> None:
        payload = {
            "document": record.model_dump(mode="json"),
            "extraction_method": result.extraction.extraction_method.value if result.extraction else None,
            "chunk_count": len(result.chunks),
            "quality": result.quality.get("statistics", {}),
            "dry_run": result.dry_run,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        processed_dir = self.documents.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        path = processed_dir / f"{record.document_id}_v{record.document_version}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    return value
