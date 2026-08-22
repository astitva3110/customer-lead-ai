"""Retrieval-oriented document models derived from canonical source-of-truth."""

from pydantic import BaseModel, Field

from app.kb.enums import (
    DocumentType,
    ExtractionMethod,
    RetrievalEligibilityStatus,
    RetrievalExclusionReason,
    RetrievalReviewReason,
    SourceType,
)
from app.kb.models.structured_content import DocumentContent, OcrMetadata


class RetrievalProvenance(BaseModel):
    """Source extraction provenance retained for downstream chunking."""

    ocr: OcrMetadata | None = None


class StructureRecoveryStats(BaseModel):
    """Deterministic structure recovery metrics for one document."""

    recovered_headings: int = 0
    recovered_clauses: int = 0
    recovered_sections: int = 0
    giant_paragraphs_before: int = 0
    giant_paragraphs_after: int = 0
    faq_pairs_detected: int = 0


class RetrievalEligibilityDecision(BaseModel):
    """Retrieval eligibility outcome for one canonical document."""

    status: RetrievalEligibilityStatus
    document_type: DocumentType
    exclusion_reason: RetrievalExclusionReason | None = None
    review_reason: RetrievalReviewReason | None = None
    notes: str | None = None

    @classmethod
    def eligible(cls, *, document_type: DocumentType) -> "RetrievalEligibilityDecision":
        return cls(status=RetrievalEligibilityStatus.ELIGIBLE, document_type=document_type)

    @classmethod
    def excluded(
        cls,
        reason: RetrievalExclusionReason,
        *,
        document_type: DocumentType,
    ) -> "RetrievalEligibilityDecision":
        return cls(
            status=RetrievalEligibilityStatus.EXCLUDED,
            document_type=document_type,
            exclusion_reason=reason,
        )

    @classmethod
    def review(
        cls,
        reason: RetrievalReviewReason,
        *,
        document_type: DocumentType,
        notes: str | None = None,
    ) -> "RetrievalEligibilityDecision":
        return cls(
            status=RetrievalEligibilityStatus.REVIEW,
            document_type=document_type,
            review_reason=reason,
            notes=notes,
        )


class RetrievalDocument(BaseModel):
    """
    Derived retrieval representation for downstream chunking.

    Contains only fields required for semantic/structural chunking.
    Pipeline, cleaning, validation, and audit metadata are excluded.
    """

    document_id: str
    document_version: int = Field(ge=1)
    kb_dataset_version: str
    title: str
    canonical_url: str
    source_url: str
    website: str
    source_type: SourceType
    extraction_method: ExtractionMethod
    document_type: DocumentType
    eligibility_status: RetrievalEligibilityStatus = RetrievalEligibilityStatus.ELIGIBLE
    exclusion_reason: RetrievalExclusionReason | None = None
    review_reason: RetrievalReviewReason | None = None
    language: str = "en"
    content_hash: str = Field(description="Document content hash for sync/manifest only")
    structured_content: DocumentContent
    retrieval_text: str = Field(description="Compact hierarchical text for chunking preview")
    provenance: RetrievalProvenance = Field(default_factory=RetrievalProvenance)
    structure_recovery: StructureRecoveryStats = Field(default_factory=StructureRecoveryStats)
