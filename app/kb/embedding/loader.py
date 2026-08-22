"""Load frozen production chunks from manifest — read only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.chunking.storage import chunk_file_path, chunk_manifest_path
from app.kb.embedding.validation import validate_production_chunk


@dataclass
class ManifestLoadResult:
    manifest_path: Path
    manifest_count: int
    chunks: list[ProductionChunkRecord]
    chunk_ids: list[str]


class ChunkLoader:
    """Read-only loader for frozen EMBED_READY production chunks."""

    def __init__(self, chunks_dir: Path, embedding_input_manifest: str) -> None:
        self.chunks_dir = chunks_dir
        self.embedding_input_manifest = embedding_input_manifest

    @property
    def manifest_path(self) -> Path:
        return chunk_manifest_path(self.chunks_dir, self.embedding_input_manifest)

    def load_manifest_entries(self) -> list[dict]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Production manifest not found: {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("chunks")
        if not isinstance(entries, list):
            raise ValueError("Manifest missing 'chunks' list")
        return entries

    def load_all(self) -> ManifestLoadResult:
        entries = self.load_manifest_entries()
        manifest_count = len(entries)
        chunks: list[ProductionChunkRecord] = []
        chunk_ids: list[str] = []
        seen_ids: set[str] = set()

        for entry in entries:
            chunk_id = entry.get("chunk_id")
            if not chunk_id:
                raise ValueError("Manifest entry missing chunk_id")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in manifest: {chunk_id}")
            seen_ids.add(chunk_id)

            website = entry.get("website")
            document_id = entry.get("document_id")
            if not website or not document_id:
                raise ValueError(f"Manifest entry missing website/document_id for chunk_id={chunk_id}")

            path = chunk_file_path(
                self.chunks_dir,
                self.embedding_input_manifest,
                website,
                document_id,
                chunk_id,
            )
            if not path.exists():
                raise FileNotFoundError(f"Manifest references missing chunk file: {path}")

            chunk = TypeAdapter(ProductionChunkRecord).validate_python(
                json.loads(path.read_text(encoding="utf-8"))
            )
            validate_production_chunk(chunk)
            chunks.append(chunk)
            chunk_ids.append(chunk_id)

        if manifest_count != len(chunks):
            raise ValueError(
                f"Manifest count mismatch: manifest={manifest_count}, loaded={len(chunks)}"
            )

        return ManifestLoadResult(
            manifest_path=self.manifest_path,
            manifest_count=manifest_count,
            chunks=chunks,
            chunk_ids=sorted(chunk_ids),
        )

    def select_deterministic(self, chunks: list[ProductionChunkRecord], count: int) -> list[ProductionChunkRecord]:
        ordered = sorted(chunks, key=lambda c: c.chunk_id)
        if len(ordered) < count:
            raise ValueError(f"Requested {count} chunks but only {len(ordered)} available")
        return ordered[:count]
