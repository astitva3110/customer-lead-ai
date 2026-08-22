"""Infrastructure adapters for IngestionService protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.kb.chunking.config import ChunkingConfig
from app.kb.ingestion.chunking import chunk_document
from app.kb.ingestion.extraction.base import DocumentExtractor, get_extractor
from app.kb.ingestion.models import DocumentRecord, ExtractionResult, Phase12ChunkRecord
from app.kb.ingestion.structure import build_structured_document
from app.kb.models.structured_content import DocumentContent


class DefaultExtractorFactory:
    def resolve(self, path: Path) -> DocumentExtractor:
        return get_extractor(path)


class StructuredContentCleaner:
    def clean(self, extraction: ExtractionResult) -> tuple[DocumentContent, Any]:
        return build_structured_document(extraction)


class Phase12DocumentChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config

    def chunk(
        self,
        *,
        record: DocumentRecord,
        extraction: ExtractionResult,
        structured: DocumentContent,
    ) -> tuple[list[Phase12ChunkRecord], list[dict[str, Any]]]:
        return chunk_document(
            record=record,
            extraction=extraction,
            structured=structured,
            config=self.config,
        )
