"""Immutable document file storage."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.kb.enums import DocumentType, DocumentUploadStatus
from app.kb.ingestion.models import DocumentRecord
from app.kb.ingestion.validation import ValidationResult, validate_upload_file


class DocumentStorage:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.root = base_dir or settings.documents_dir
        root = self.root
        self.uploaded_dir = root / "uploaded"
        self.processed_dir = root / "processed"
        self.failed_dir = root / "failed"
        self.registry_path = root / "registry.json"
        for directory in (self.uploaded_dir, self.processed_dir, self.failed_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, DocumentRecord] = self._load_registry()

    def _load_registry(self) -> dict[str, DocumentRecord]:
        if not self.registry_path.exists():
            return {}
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {item["document_id"]: DocumentRecord.model_validate(item) for item in raw.get("documents", [])}

    def _save_registry(self) -> None:
        payload = {"documents": [record.model_dump(mode="json") for record in self._registry.values()]}
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        for record in self._registry.values():
            if record.file_hash == file_hash:
                return record
        return None

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._registry.get(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self._registry.values(), key=lambda item: item.created_at)

    def _document_id_for_hash(self, file_hash: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"phase12-document:{file_hash}"))

    def store_upload(
        self,
        source_path: Path,
        *,
        document_type: DocumentType,
        language: str = "en",
        source: str = "upload",
    ) -> tuple[DocumentRecord, ValidationResult, bool]:
        """Copy file to uploaded/ if new. Returns (record, validation, is_duplicate)."""
        validation = validate_upload_file(source_path)
        if not validation.ok:
            return (
                DocumentRecord(
                    document_id="",
                    filename=source_path.name,
                    mime_type=validation.mime_type,
                    file_hash=validation.file_hash,
                    document_type=document_type,
                    language=language,
                    source=source,
                    created_at=datetime.now(timezone.utc),
                    status=DocumentUploadStatus.FAILED,
                    error=validation.error,
                ),
                validation,
                False,
            )

        existing = self.find_by_hash(validation.file_hash)
        if existing:
            return existing, validation, True

        document_id = self._document_id_for_hash(validation.file_hash)
        target_name = f"{document_id}{validation.extension}"
        target_path = self.uploaded_dir / target_name
        if target_path.exists():
            # Immutable store — never overwrite.
            pass
        else:
            shutil.copy2(source_path, target_path)

        record = DocumentRecord(
            document_id=document_id,
            document_version=1,
            filename=source_path.name,
            mime_type=validation.mime_type,
            file_hash=validation.file_hash,
            document_type=document_type,
            language=language,
            source=source,
            created_at=datetime.now(timezone.utc),
            status=DocumentUploadStatus.UPLOADED,
            storage_path=str(target_path),
            title=source_path.stem,
        )
        self._registry[document_id] = record
        self._save_registry()
        self.copy_original(document_id, target_path)
        return record, validation, False

    def document_dir(self, document_id: str) -> Path:
        path = self.root / document_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def copy_original(self, document_id: str, source: Path) -> None:
        dest = self.document_dir(document_id) / f"original{source.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(source, dest)

    def write_json(self, document_id: str, relative_path: str, payload: dict) -> None:
        path = self.document_dir(document_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def mark_status(
        self,
        document_id: str,
        status: DocumentUploadStatus,
        *,
        error: str | None = None,
        pages: int | None = None,
        chunks: int | None = None,
        embedded: int | None = None,
    ) -> None:
        record = self._registry[document_id]
        record.status = status
        if error is not None:
            record.error = error
        if pages is not None:
            record.pages = pages
        if chunks is not None:
            record.chunks = chunks
        if embedded is not None:
            record.embedded = embedded
        self._save_registry()

    def bump_version(self, document_id: str, new_hash: str, new_path: Path) -> DocumentRecord:
        record = self._registry[document_id]
        record.document_version += 1
        record.file_hash = new_hash
        record.storage_path = str(new_path)
        record.status = DocumentUploadStatus.UPLOADED
        record.error = None
        self._save_registry()
        return record
