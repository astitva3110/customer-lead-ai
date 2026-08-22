from typing import Protocol

from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord


class VectorRepository(Protocol):
    def ensure_schema(self) -> None: ...

    def document_vectors_exist(self, document_id: str, document_version: int) -> bool: ...

    def upsert_chunks(
        self,
        *,
        record: DocumentRecord,
        chunks: list[Phase12ChunkRecord],
        embeddings: list[list[float]],
        title: str,
        force: bool = False,
    ) -> int: ...
