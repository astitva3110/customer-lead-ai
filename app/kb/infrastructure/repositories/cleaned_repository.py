import json
from pathlib import Path

from pydantic import TypeAdapter

from app.kb.enums import SourceType
from app.kb.hashing import hash_filename
from app.kb.models.raw import CleanedArtifact
from app.kb.infrastructure.repositories.paths import website_source_dir


class CleanedRepository:
    """Storage for CLEANED stage artifacts (structure only in Phase 1)."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, website: str, source_type: SourceType, content_hash: str) -> Path:
        filename = hash_filename(content_hash)
        return website_source_dir(self.base_dir, website, source_type) / filename

    def write(self, artifact: CleanedArtifact) -> Path:
        path = self.artifact_path(artifact.website, artifact.source_type, artifact.content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path
        path.write_text(
            json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def read_path(self, path: Path) -> CleanedArtifact:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(CleanedArtifact).validate_python(data)

    def exists(self, website: str, source_type: SourceType, content_hash: str) -> bool:
        return self.artifact_path(website, source_type, content_hash).exists()

    def get_by_hash(self, website: str, source_type: SourceType, content_hash: str) -> CleanedArtifact | None:
        path = self.artifact_path(website, source_type, content_hash)
        if not path.exists():
            return None
        return self.read_path(path)
