"""Version-agnostic retrieval evaluation configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from app.config import settings

RetrievalMode = Literal["pgvector", "in-memory"]


@dataclass(frozen=True)
class RetrievalConfig:
    corpus_version: str
    vector_table: str | None
    embedding_model: str
    embedding_revision: str
    embedding_version: str
    kb_dataset_version: str
    chunking_algorithm_version: str
    embedding_input_manifest: str
    top_k: int = 100
    mode: RetrievalMode = "pgvector"
    chunks_path: str | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> RetrievalConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalConfig:
        return cls(
            corpus_version=data["corpus_version"],
            vector_table=data.get("vector_table"),
            embedding_model=data.get("embedding_model", settings.embedding_model),
            embedding_revision=data.get("embedding_model_revision", settings.embedding_model_revision),
            embedding_version=data["embedding_version"],
            kb_dataset_version=data["kb_dataset_version"],
            chunking_algorithm_version=data["chunking_algorithm_version"],
            embedding_input_manifest=data["embedding_input_manifest"],
            top_k=int(data.get("top_k", 100)),
            mode=data.get("mode", "pgvector"),
            chunks_path=data.get("chunks_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "vector_table": self.vector_table,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "embedding_version": self.embedding_version,
            "kb_dataset_version": self.kb_dataset_version,
            "chunking_algorithm_version": self.chunking_algorithm_version,
            "embedding_input_manifest": self.embedding_input_manifest,
            "top_k": self.top_k,
            "mode": self.mode,
            "chunks_path": self.chunks_path,
        }


def load_questions(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("questions", [])
