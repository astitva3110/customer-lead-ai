"""Chunk-set hard gate used by IngestionService before embedding."""

from __future__ import annotations

from app.kb.ingestion.quality import validate_chunk_set

__all__ = ["validate_chunk_set"]
