"""Downstream contract for future chunking pipeline consumption."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.kb.enums import ExtractionMethod, ProcessingStatus, SourceType
from app.kb.models.structured_content import DocumentContent


class DownstreamDocument(BaseModel):
    """
    Guaranteed schema for documents consumed by the chunking pipeline.

    Only ACTIVE and VALIDATED target-domain documents should be exported
    using this contract.
    """

    document_id: str
    document_version: int = Field(description="Canonical document version number")
    kb_dataset_version: str = Field(description="Frozen KB snapshot version")
    title: str
    canonical_url: str
    source_url: str
    website: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    processing_status: ProcessingStatus
    content_hash: str
    structured_content: DocumentContent
    plain_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    links: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "en"
    scraped_at: datetime
    updated_at: datetime

    @classmethod
    def from_canonical(cls, document, *, kb_dataset_version: str) -> "DownstreamDocument":
        from app.kb.models.canonical import CanonicalDocument

        if not isinstance(document, CanonicalDocument):
            raise TypeError("expected CanonicalDocument")

        return cls(
            document_id=document.document_id,
            document_version=document.version,
            kb_dataset_version=kb_dataset_version,
            title=document.title,
            canonical_url=document.canonical_url,
            source_url=document.source_url,
            website=document.website,
            source_type=document.source_type,
            extraction_method=document.extraction_method,
            processing_status=document.processing_status,
            content_hash=document.content_hash,
            structured_content=document.structured_content,
            plain_text=document.plain_text or "",
            metadata=document.metadata,
            links=[link.model_dump(mode="json") for link in document.links],
            language=document.language,
            scraped_at=document.scraped_at,
            updated_at=document.updated_at,
        )


def downstream_eligible(document) -> bool:
    """Documents eligible for downstream chunking export."""
    from app.kb.enums import ProcessingStatus
    from app.kb.policy.domain_policy import is_target_website

    return (
        is_target_website(document.website)
        and document.processing_status in (ProcessingStatus.ACTIVE, ProcessingStatus.VALIDATED)
        and not document.metadata.get("excluded", False)
    )
