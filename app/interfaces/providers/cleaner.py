from typing import Any, Protocol

from app.kb.ingestion.models import ExtractionResult
from app.kb.models.structured_content import DocumentContent


class DocumentCleaner(Protocol):
    def clean(self, extraction: ExtractionResult) -> tuple[DocumentContent, Any]: ...
