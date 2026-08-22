import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.kb.enums import SourceType
from app.kb.hashing import hash_filename, hash_raw_payload
from app.kb.models.raw import RawArtifact
from app.kb.infrastructure.repositories.paths import website_source_dir


@dataclass
class RawWriteResult:
    artifact: RawArtifact
    path: Path
    created: bool
    duplicate: bool


class RawRepository:
    """Append-only immutable RAW artifact storage."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self, website: str) -> Path:
        return self.base_dir / website / "_index.json"

    def _load_index(self, website: str) -> dict[str, Any]:
        index_path = self._index_path(website)
        if not index_path.exists():
            return {"documents": {}}
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _save_index(self, website: str, index: dict[str, Any]) -> None:
        index_path = self._index_path(website)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def artifact_path(self, website: str, source_type: SourceType, content_hash: str) -> Path:
        filename = hash_filename(content_hash)
        return website_source_dir(self.base_dir, website, source_type) / filename

    def exists(self, website: str, source_type: SourceType, content_hash: str) -> bool:
        return self.artifact_path(website, source_type, content_hash).exists()

    def write(self, artifact: RawArtifact) -> RawWriteResult:
        """
        Write a RAW artifact immutably.

        If an artifact with the same content hash already exists, returns the
        existing record without overwriting.
        """
        payload = artifact.model_dump(mode="json")
        content_hash_value = artifact.content_hash or hash_raw_payload(payload)
        artifact = artifact.model_copy(update={"content_hash": content_hash_value})

        path = self.artifact_path(artifact.website, artifact.source_type, content_hash_value)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = self.read_path(path)
            self._update_index(existing, created=False)
            return RawWriteResult(artifact=existing, path=path, created=False, duplicate=True)

        path.write_text(
            json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._update_index(artifact, created=True)
        return RawWriteResult(artifact=artifact, path=path, created=True, duplicate=False)

    def _update_index(self, artifact: RawArtifact, *, created: bool) -> None:
        index = self._load_index(artifact.website)
        documents: dict[str, Any] = index.setdefault("documents", {})
        key = artifact.canonical_url
        entry = documents.get(key, {"versions": []})

        versions: list[dict[str, Any]] = entry.setdefault("versions", [])
        known_hashes = {v["content_hash"] for v in versions}
        if artifact.content_hash not in known_hashes:
            versions.append(
                {
                    "content_hash": artifact.content_hash,
                    "artifact_path": str(
                        self.artifact_path(artifact.website, artifact.source_type, artifact.content_hash)
                    ),
                    "scraped_at": artifact.scraped_at.isoformat(),
                    "created": created,
                }
            )

        entry["latest_content_hash"] = artifact.content_hash
        entry["source_type"] = artifact.source_type
        entry["title"] = artifact.title
        documents[key] = entry
        self._save_index(artifact.website, index)

    def read_path(self, path: Path) -> RawArtifact:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(RawArtifact).validate_python(data)

    def get_by_hash(self, website: str, source_type: SourceType, content_hash: str) -> RawArtifact | None:
        path = self.artifact_path(website, source_type, content_hash)
        if not path.exists():
            return None
        return self.read_path(path)

    def get_latest_for_url(self, website: str, canonical_url: str) -> RawArtifact | None:
        index = self._load_index(website)
        entry = index.get("documents", {}).get(canonical_url)
        if not entry:
            return None
        latest_hash = entry.get("latest_content_hash")
        source_type = SourceType(entry["source_type"])
        if not latest_hash:
            return None
        return self.get_by_hash(website, source_type, latest_hash)

    def is_duplicate_content(self, website: str, canonical_url: str, content_hash: str) -> bool:
        index = self._load_index(website)
        entry = index.get("documents", {}).get(canonical_url)
        if not entry:
            return False
        return content_hash in {v["content_hash"] for v in entry.get("versions", [])}
