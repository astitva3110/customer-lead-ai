from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.kb.enums import ExtractionMethod, ProcessingStatus, SourceType
from app.kb.models.structured_content import DocumentContent


class DocumentLink(BaseModel):
    url: str
    text: str | None = None
    internal: bool = True


class CanonicalDocument(BaseModel):
    document_id: str
    website: str
    source_url: str
    canonical_url: str
    title: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    language: str = "en"
    version: int = Field(default=1, ge=1)
    is_current: bool = True
    content_hash: str
    processing_status: ProcessingStatus = ProcessingStatus.EXTRACTED
    scraped_at: datetime
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured_content: DocumentContent
    plain_text: str | None = None
    links: list[DocumentLink] = Field(default_factory=list)

    @field_validator("source_type", mode="before")
    @classmethod
    def validate_source_type(cls, value: str | SourceType) -> SourceType:
        if isinstance(value, SourceType):
            return value
        return SourceType(value)

    @field_validator("extraction_method", mode="before")
    @classmethod
    def validate_extraction_method(cls, value: str | ExtractionMethod) -> ExtractionMethod:
        if isinstance(value, ExtractionMethod):
            return value
        return ExtractionMethod(value)

    @field_validator("processing_status", mode="before")
    @classmethod
    def validate_processing_status(cls, value: str | ProcessingStatus) -> ProcessingStatus:
        if isinstance(value, ProcessingStatus):
            return value
        return ProcessingStatus(value)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not value.startswith("sha256:") or len(value) != 71:
            raise ValueError("content_hash must be sha256:<64 hex chars>")
        hex_part = value.removeprefix("sha256:")
        if not all(c in "0123456789abcdef" for c in hex_part):
            raise ValueError("content_hash must contain valid hex digest")
        return value


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
