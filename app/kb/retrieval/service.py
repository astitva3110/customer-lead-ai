"""Version-agnostic retrieval service."""

from __future__ import annotations

from app.kb.evaluation.models import RankedHit
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import RetrievalBackend, build_backend
from app.kb.ingestion.models import Phase12ChunkRecord


class RetrievalService:
    """Thin facade over retrieval backends — corpus version is configuration."""

    def __init__(self, backend: RetrievalBackend) -> None:
        self._backend = backend

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        *,
        chunks: list[Phase12ChunkRecord] | None = None,
        device: str | None = None,
    ) -> RetrievalService:
        return cls(build_backend(config, chunks=chunks, device=device))

    def search(self, query: str, *, top_k: int | None = None) -> list[RankedHit]:
        effective_top_k = top_k if top_k is not None else 100
        return self._backend.search(query, top_k=effective_top_k)
