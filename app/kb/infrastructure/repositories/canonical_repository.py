import json
from pathlib import Path

from pydantic import TypeAdapter

from app.kb.models.canonical import CanonicalDocument
from app.kb.infrastructure.repositories.paths import canonical_website_dir


class CanonicalRepository:
    """Filesystem storage for canonical documents."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def document_dir(self, website: str, document_id: str) -> Path:
        return canonical_website_dir(self.base_dir, website) / document_id

    def version_path(self, website: str, document_id: str, version: int) -> Path:
        return self.document_dir(website, document_id) / f"v{version}.json"

    def current_path(self, website: str, document_id: str) -> Path:
        return self.document_dir(website, document_id) / "current.json"

    def write(self, document: CanonicalDocument) -> Path:
        doc_dir = self.document_dir(document.website, document.document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        version_path = self.version_path(document.website, document.document_id, document.version)
        payload = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2)

        if not version_path.exists():
            version_path.write_text(payload, encoding="utf-8")

        if document.is_current:
            self.current_path(document.website, document.document_id).write_text(payload, encoding="utf-8")

        return version_path

    def read_current(self, website: str, document_id: str) -> CanonicalDocument | None:
        path = self.current_path(website, document_id)
        if not path.exists():
            return None
        return self.read_path(path)

    def read_path(self, path: Path) -> CanonicalDocument:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(CanonicalDocument).validate_python(data)

    def read_version(self, website: str, document_id: str, version: int) -> CanonicalDocument | None:
        path = self.version_path(website, document_id, version)
        if not path.exists():
            return None
        return self.read_path(path)
