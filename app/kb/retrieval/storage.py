"""Filesystem storage for retrieval-prepared documents."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from app.kb.retrieval.models import RetrievalDocument
from app.kb.storage.paths import retrieval_dataset_dir, retrieval_document_path


class RetrievalStore:
    """Write/read derived retrieval documents without touching canonical storage."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def document_path(self, kb_dataset_version: str, website: str, document_id: str) -> Path:
        return retrieval_document_path(self.base_dir, kb_dataset_version, website, document_id)

    def dataset_dir(self, kb_dataset_version: str) -> Path:
        return retrieval_dataset_dir(self.base_dir, kb_dataset_version)

    def write(self, document: RetrievalDocument) -> Path:
        path = self.document_path(
            document.kb_dataset_version,
            document.website,
            document.document_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")
        return path

    def read(self, kb_dataset_version: str, website: str, document_id: str) -> RetrievalDocument | None:
        path = self.document_path(kb_dataset_version, website, document_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(RetrievalDocument).validate_python(data)

    def delete(self, kb_dataset_version: str, website: str, document_id: str) -> bool:
        path = self.document_path(kb_dataset_version, website, document_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def delete_excluded_documents(self, kb_dataset_version: str, excluded_entries: list[dict]) -> int:
        """Remove retrieval files for excluded documents (storage invariant)."""
        removed = 0
        for entry in excluded_entries:
            if self.delete(kb_dataset_version, entry["website"], entry["document_id"]):
                removed += 1
        return removed

    def manifest_path(self, kb_dataset_version: str) -> Path:
        return self.dataset_dir(kb_dataset_version) / "manifest.json"

    def eligibility_manifest_path(self, kb_dataset_version: str) -> Path:
        return self.dataset_dir(kb_dataset_version) / "eligibility.json"

    def write_manifest(self, kb_dataset_version: str, manifest: dict) -> Path:
        path = self.manifest_path(kb_dataset_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_eligibility_manifest(self, kb_dataset_version: str, manifest: dict) -> Path:
        path = self.eligibility_manifest_path(kb_dataset_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
