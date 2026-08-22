"""Canonical document integrity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.kb.enums import ExtractionMethod, ProcessingStatus, SourceType
from app.kb.integrity.consistency import check_content_consistency, find_empty_nodes
from app.kb.models.canonical import CanonicalDocument
from app.kb.policy.domain_policy import is_target_website, website_policy
from app.kb.url_normalizer import make_document_id, normalize_url

PRODUCTION_STATUSES = frozenset({ProcessingStatus.ACTIVE, ProcessingStatus.VALIDATED})


@dataclass
class IntegrityIssue:
    document_id: str
    website: str
    url: str
    category: str
    message: str
    severity: str = "ERROR"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "website": self.website,
            "url": self.url,
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class DocumentIntegrityResult:
    document_id: str
    website: str
    url: str
    processing_status: str
    passed: bool
    issues: list[IntegrityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "website": self.website,
            "url": self.url,
            "processing_status": self.processing_status,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _issue(doc: CanonicalDocument, category: str, message: str, severity: str = "ERROR") -> IntegrityIssue:
    return IntegrityIssue(
        document_id=doc.document_id,
        website=doc.website,
        url=doc.canonical_url,
        category=category,
        message=message,
        severity=severity,
    )


def check_document_integrity(document: CanonicalDocument) -> DocumentIntegrityResult:
    issues: list[IntegrityIssue] = []
    status = document.processing_status

    # Identity
    if not document.document_id:
        issues.append(_issue(document, "identity", "missing document_id"))
    else:
        expected_id = make_document_id(document.website, document.canonical_url)
        if document.document_id != expected_id:
            issues.append(
                _issue(
                    document,
                    "identity",
                    f"document_id not stable: expected {expected_id}, got {document.document_id}",
                )
            )

    if not document.canonical_url:
        issues.append(_issue(document, "identity", "missing canonical_url"))
    if not document.source_url:
        issues.append(_issue(document, "identity", "missing source_url"))

    if document.website not in {document.website.lower()}:
        issues.append(_issue(document, "identity", "website should be lowercase"))

    try:
        SourceType(document.source_type)
    except ValueError:
        issues.append(_issue(document, "identity", f"invalid source_type: {document.source_type}"))

    try:
        ExtractionMethod(document.extraction_method)
    except ValueError:
        issues.append(_issue(document, "identity", f"invalid extraction_method: {document.extraction_method}"))

    # Version
    if document.version < 1:
        issues.append(_issue(document, "version", "invalid version number"))
    if not document.content_hash or not document.content_hash.startswith("sha256:"):
        issues.append(_issue(document, "version", "invalid content_hash"))
    if not document.is_current and status in PRODUCTION_STATUSES:
        issues.append(_issue(document, "version", "production document is not marked is_current"))

    # Domain / exclusion policy
    if status in PRODUCTION_STATUSES:
        if not is_target_website(document.website):
            issues.append(_issue(document, "domain", "NON_TARGET website cannot be ACTIVE/VALIDATED"))
        if document.metadata.get("excluded"):
            issues.append(_issue(document, "exclusion", "excluded document cannot be ACTIVE/VALIDATED"))

    if status == ProcessingStatus.EXCLUDED:
        if not document.metadata.get("excluded"):
            issues.append(_issue(document, "exclusion", "EXCLUDED status but excluded flag not set"))
        if not document.metadata.get("exclusion_reason"):
            issues.append(_issue(document, "exclusion", "EXCLUDED status missing exclusion_reason"))

    if status == ProcessingStatus.FAILED:
        if is_target_website(document.website):
            issues.append(
                _issue(
                    document,
                    "status",
                    "FAILED target-domain document must be resolved before freeze",
                    severity="ERROR",
                )
            )

    # Content checks for production documents
    if status in PRODUCTION_STATUSES:
        if not document.title.strip():
            issues.append(_issue(document, "content", "missing title", severity="WARNING"))

        if not (document.plain_text or "").strip():
            issues.append(_issue(document, "content", "plain_text is empty"))

        if document.structured_content is None:
            issues.append(_issue(document, "content", "missing structured_content"))
        else:
            for msg in find_empty_nodes(document.structured_content):
                issues.append(_issue(document, "structure", msg, severity="WARNING"))
            for msg in check_content_consistency(document):
                severity = "WARNING" if "ratio" in msg or "coverage" in msg else "ERROR"
                issues.append(_issue(document, "consistency", msg, severity=severity))

        # PDF integrity
        if document.source_type in (SourceType.PDF, SourceType.SCANNED_PDF):
            pages = document.structured_content.pages or []
            if not pages:
                issues.append(_issue(document, "pdf", "pdf document missing page structure", severity="WARNING"))
            else:
                numbers = [p.page_number for p in pages]
                if numbers != sorted(numbers):
                    issues.append(_issue(document, "pdf", "pdf pages out of order"))
                if len(set(numbers)) != len(numbers):
                    issues.append(_issue(document, "pdf", "duplicate pdf page numbers"))
                if any(n < 1 for n in numbers):
                    issues.append(_issue(document, "pdf", "invalid pdf page numbers"))

        # OCR consistency (no confidence required)
        if document.extraction_method == ExtractionMethod.OCR:
            if document.source_type != SourceType.SCANNED_PDF:
                issues.append(
                    _issue(
                        document,
                        "ocr",
                        "OCR extraction_method should use scanned_pdf source_type",
                        severity="WARNING",
                    )
                )
        elif document.source_type == SourceType.SCANNED_PDF and document.extraction_method != ExtractionMethod.OCR:
            issues.append(_issue(document, "ocr", "scanned_pdf should use OCR extraction_method", severity="WARNING"))

    errors = [i for i in issues if i.severity == "ERROR"]
    passed = not errors
    if status in PRODUCTION_STATUSES:
        passed = not errors

    return DocumentIntegrityResult(
        document_id=document.document_id,
        website=document.website,
        url=document.canonical_url,
        processing_status=status.value,
        passed=passed,
        issues=issues,
    )
