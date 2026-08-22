from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.kb.enums import ExtractionMethod, SourceType
from app.kb.models.structured_content import OcrMetadata, PageContent


class RawArtifact(BaseModel):
    """Immutable scraped document stored under data/raw/."""

    url: str
    canonical_url: str
    website: str
    title: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    content: str
    scraped_at: datetime
    content_hash: str
    crawl_root_url: str | None = None
    pages: list[PageContent] | None = None
    ocr: OcrMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CleanedArtifact(BaseModel):
    """Cleaned document stored under data/cleaned/."""

    raw_content_hash: str
    url: str
    canonical_url: str
    website: str
    title: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    content: str
    cleaned_at: datetime
    content_hash: str
    pages: list[PageContent] | None = None
    ocr: OcrMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
