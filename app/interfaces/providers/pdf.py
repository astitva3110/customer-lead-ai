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


class ExtractorFactory(Protocol):
    def resolve(self, path: Path) -> DocumentExtractor: ...
