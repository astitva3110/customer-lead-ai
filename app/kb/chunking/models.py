"""Chunk record models for dry-run and future production indexing."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.kb.enums import DocumentType, ExtractionMethod, SourceType


class SplitMethod(StrEnum):
    """How a chunk boundary was determined."""

    SEMANTIC = "semantic"
    SUBSECTION = "subsection"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CLAUSE = "clause"
    LIST = "list"
    TABLE = "table"
    FAQ_PAIR = "faq_pair"
    SENTENCE = "sentence"
    HARD_TOKEN_FALLBACK = "hard_token_fallback"


class ChunkProvenance(BaseModel):
    """Provenance metadata kept separate from embeddable content."""

    content_hash: str = Field(description="Parent retrieval document content hash")
    source_block_path: list[str] = Field(default_factory=list)
    ocr_engine: str | None = None
    ocr_dpi: int | None = None
    ocr_language: str | None = None
    ocr_page_confidence: float | None = None


class ChunkRecord(BaseModel):
    """One deterministic semantic chunk derived from a retrieval document."""

    chunk_id: str
    document_id: str
    document_version: int = Field(ge=1)
    kb_dataset_version: str
    chunk_index: int = Field(ge=0)
    website: str
    document_type: DocumentType
    title: str
    section_path: list[str] = Field(default_factory=list)
    content: str
    token_count: int = Field(ge=0)
    split_method: SplitMethod
    source_url: str
    canonical_url: str
    page_number: int | None = None
    source_type: SourceType
    extraction_method: ExtractionMethod
    overlap_tokens: int = 0
    parent_chunk_id: str | None = None
    provenance: ChunkProvenance
    language: str = "en"


class ProductionChunkRecord(ChunkRecord):
    """Production embedding manifest record — always EMBED_READY."""

    eligibility_status: str = "EMBED_READY"
