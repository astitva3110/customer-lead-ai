"""Phase 12 document-first ingestion models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.kb.enums import DocumentType, DocumentUploadStatus, ExtractionMethod, SourceType


class ContentUnitType(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    CLAUSE = "clause"


class ExtractedUnit(BaseModel):
    document_id: str
    document_version: int = Field(ge=1)
    page_number: int | None = None
    text: str
    content_type: ContentUnitType = ContentUnitType.PARAGRAPH
    extraction_method: ExtractionMethod
    ocr_used: bool = False
    ocr_confidence: float | None = None
    heading_level: int | None = None
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    list_items: list[str] = Field(default_factory=list)
    list_ordered: bool = False


class ExtractionResult(BaseModel):
    document_id: str
    document_version: int = Field(ge=1)
    title: str
    units: list[ExtractedUnit] = Field(default_factory=list)
    extraction_method: ExtractionMethod
    ocr_used: bool = False
    native_page_count: int = 0
    ocr_page_count: int = 0
    page_count: int = 0


class DocumentRecord(BaseModel):
    document_id: str
    document_version: int = Field(ge=1, default=1)
    filename: str
    mime_type: str
    file_hash: str
    document_type: DocumentType
    language: str = "en"
    source: str = "upload"
    created_at: datetime
    status: DocumentUploadStatus = DocumentUploadStatus.UPLOADED
    storage_path: str = ""
    title: str | None = None
    error: str | None = None
    pages: int = 0
    chunks: int = 0
    embedded: int = 0


class Phase12ChunkRecord(BaseModel):
    """Versioned chunk schema for document-first ingestion."""

    chunk_id: str
    document_id: str
    document_version: int = Field(ge=1)
    document_type: DocumentType
    chunking_algorithm_version: str
    content: str
    embedding_input: str
    embedding_input_hash: str
    token_count: int = Field(ge=0)
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_section: str | None = None
    subsection: str | None = None
    content_type: str = "paragraph"
    source_file_hash: str
    extraction_method: ExtractionMethod
    ocr_used: bool = False
    created_at: datetime
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_model_revision: str = "69da0546e10ee869fbf19f3b1d6c5ac12eb48a16"
    embedding_dimension: int = 1024
    embedding_version: str = ""
    split_method: str = "semantic"
    overlap_tokens: int = 0
    suppressed: bool = False
    suppression_reason: str | None = None


class IngestionResult(BaseModel):
    document: DocumentRecord
    extraction: ExtractionResult | None = None
    chunks: list[Phase12ChunkRecord] = Field(default_factory=list)
    suppressed_chunks: list[dict[str, Any]] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    embedded: bool = False
    indexed: bool = False
    duplicate: bool = False
