from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class ReviewSeverity(StrEnum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ReviewItem:
    document_id: str
    website: str
    url: str
    title: str
    source_type: str
    issue: str
    severity: ReviewSeverity
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoilerplateMatch:
    document_id: str
    website: str
    url: str
    title: str
    pattern: str
    category: str
    location: str
    severity: ReviewSeverity
    context: str = ""


@dataclass
class DocumentAuditRecord:
    document_id: str
    website: str
    canonical_url: str
    source_url: str
    title: str
    source_type: str
    extraction_method: str
    processing_status: str
    content_hash: str
    raw_content_hash: str | None
    cleaned_content_hash: str | None
    identity: dict[str, Any] = field(default_factory=dict)
    content: dict[str, Any] = field(default_factory=dict)
    pdf: dict[str, Any] | None = None
    ocr: dict[str, Any] | None = None
    links: dict[str, Any] = field(default_factory=dict)
    content_loss: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    boilerplate_matches: list[BoilerplateMatch] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "website": self.website,
            "canonical_url": self.canonical_url,
            "source_url": self.source_url,
            "title": self.title,
            "source_type": self.source_type,
            "extraction_method": self.extraction_method,
            "processing_status": self.processing_status,
            "content_hash": self.content_hash,
            "raw_content_hash": self.raw_content_hash,
            "cleaned_content_hash": self.cleaned_content_hash,
            "identity": self.identity,
            "content": self.content,
            "pdf": self.pdf,
            "ocr": self.ocr,
            "links": self.links,
            "content_loss": self.content_loss,
            "structure": self.structure,
            "boilerplate_matches": [
                {
                    "pattern": m.pattern,
                    "category": m.category,
                    "location": m.location,
                    "severity": m.severity,
                    "context": m.context,
                }
                for m in self.boilerplate_matches
            ],
            "review_items": [
                {
                    "issue": r.issue,
                    "severity": r.severity,
                    "context": r.context,
                }
                for r in self.review_items
            ],
            "validation": self.validation,
        }


@dataclass
class AuditSummary:
    generated_at: str
    raw_count: int
    cleaned_count: int
    canonical_count: int
    by_website: dict[str, int]
    by_source_type: dict[str, int]
    by_extraction_method: dict[str, int]
    by_processing_status: dict[str, int]
    by_website_source_type: dict[str, dict[str, int]]
    validated_count: int
    warning_count: int
    failed_count: int
    excluded_count: int
    active_count: int
    duplicate_groups: int
    boilerplate_document_count: int
    high_content_loss_count: int
    moderate_content_loss_count: int
    ocr_issue_count: int
    review_critical: int
    review_warning: int
    review_info: int
    human_review_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "raw_count": self.raw_count,
            "cleaned_count": self.cleaned_count,
            "canonical_count": self.canonical_count,
            "by_website": self.by_website,
            "by_source_type": self.by_source_type,
            "by_extraction_method": self.by_extraction_method,
            "by_processing_status": self.by_processing_status,
            "by_website_source_type": self.by_website_source_type,
            "validated_count": self.validated_count,
            "warning_count": self.warning_count,
            "failed_count": self.failed_count,
            "excluded_count": self.excluded_count,
            "active_count": self.active_count,
            "duplicate_groups": self.duplicate_groups,
            "boilerplate_document_count": self.boilerplate_document_count,
            "high_content_loss_count": self.high_content_loss_count,
            "moderate_content_loss_count": self.moderate_content_loss_count,
            "ocr_issue_count": self.ocr_issue_count,
            "review_critical": self.review_critical,
            "review_warning": self.review_warning,
            "review_info": self.review_info,
            "human_review_count": self.human_review_count,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
