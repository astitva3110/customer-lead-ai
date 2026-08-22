from typing import Any, Protocol

from app.kb.ingestion.models import DocumentRecord, ExtractionResult, Phase12ChunkRecord
from app.kb.models.structured_content import DocumentContent


class DocumentChunker(Protocol):
    def chunk(
        self,
        *,
        record: DocumentRecord,
        extraction: ExtractionResult,
        structured: DocumentContent,
    ) -> tuple[list[Phase12ChunkRecord], list[dict[str, Any]]]: ...
