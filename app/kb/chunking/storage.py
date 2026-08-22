"""Production chunk filesystem storage."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from app.kb.chunking.models import ProductionChunkRecord


def chunk_dataset_dir(base_dir: Path, kb_dataset_version: str) -> Path:
    return base_dir / kb_dataset_version


def chunk_manifest_path(base_dir: Path, kb_dataset_version: str) -> Path:
    return chunk_dataset_dir(base_dir, kb_dataset_version) / "manifest.json"


def review_manifest_path(base_dir: Path, kb_dataset_version: str) -> Path:
    return chunk_dataset_dir(base_dir, kb_dataset_version) / "review_manifest.json"


def chunk_file_path(base_dir: Path, kb_dataset_version: str, website: str, document_id: str, chunk_id: str) -> Path:
    return chunk_dataset_dir(base_dir, kb_dataset_version) / website / document_id / f"{chunk_id}.json"


class ChunkStore:
    """Write/read production EMBED_READY chunk records."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_production_chunk(self, chunk: ProductionChunkRecord) -> Path:
        path = chunk_file_path(
            self.base_dir,
            chunk.kb_dataset_version,
            chunk.website,
            chunk.document_id,
            chunk.chunk_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_manifest(self, kb_dataset_version: str, manifest: dict) -> Path:
        path = chunk_manifest_path(self.base_dir, kb_dataset_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_review_manifest(self, kb_dataset_version: str, manifest: dict) -> Path:
        path = review_manifest_path(self.base_dir, kb_dataset_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_production_chunk(
        self,
        kb_dataset_version: str,
        website: str,
        document_id: str,
        chunk_id: str,
    ) -> ProductionChunkRecord | None:
        path = chunk_file_path(self.base_dir, kb_dataset_version, website, document_id, chunk_id)
        if not path.exists():
            return None
        return TypeAdapter(ProductionChunkRecord).validate_python(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def list_production_chunks(self, kb_dataset_version: str) -> list[Path]:
        root = chunk_dataset_dir(self.base_dir, kb_dataset_version)
        if not root.exists():
            return []
        return sorted(root.glob("*/*/*.json"))
