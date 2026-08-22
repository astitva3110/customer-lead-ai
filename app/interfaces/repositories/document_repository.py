from pathlib import Path
from typing import Protocol

from app.kb.enums import DocumentUploadStatus
from app.kb.ingestion.models import DocumentRecord
from app.kb.ingestion.validation import ValidationResult


class DocumentRepository(Protocol):
    failed_dir: Path
    processed_dir: Path

    def store_upload(
        self,
        source_path: Path,
        *,
        document_type,
        language: str = "en",
        source: str = "upload",
    ) -> tuple[DocumentRecord, ValidationResult, bool]: ...

    def get(self, document_id: str) -> DocumentRecord | None: ...

    def mark_status(
        self,
        document_id: str,
        status: DocumentUploadStatus,
        *,
        error: str | None = None,
        pages: int | None = None,
        chunks: int | None = None,
        embedded: int | None = None,
    ) -> None: ...

    def write_json(self, document_id: str, relative_path: str, payload: dict) -> None: ...

    def copy_original(self, document_id: str, source: Path) -> None: ...
