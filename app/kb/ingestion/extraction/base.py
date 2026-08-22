"""Extraction base types and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.kb.ingestion.models import ExtractionResult


class DocumentExtractor(Protocol):
    def extract(
        self,
        path: Path,
        *,
        document_id: str,
        document_version: int,
        title: str | None = None,
        allow_ocr: bool = True,
    ) -> ExtractionResult: ...


def get_extractor(path: Path) -> DocumentExtractor:
    extension = path.suffix.lower()
    if extension == ".pdf":
        from app.kb.ingestion.extraction.pdf import PdfDocumentExtractor

        return PdfDocumentExtractor()
    if extension == ".docx":
        from app.kb.ingestion.extraction.docx import DocxDocumentExtractor

        return DocxDocumentExtractor()
    if extension == ".doc":
        from app.kb.ingestion.extraction.doc import DocDocumentExtractor

        return DocDocumentExtractor()
    raise ValueError(f"No extractor for extension {extension!r}")
