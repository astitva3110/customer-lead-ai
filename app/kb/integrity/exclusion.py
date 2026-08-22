"""Exclusion integrity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.kb.enums import ExclusionReason, ProcessingStatus
from app.kb.models.canonical import CanonicalDocument
from app.kb.policy.domain_policy import is_target_website


@dataclass
class ExclusionIntegrityReport:
    excluded_documents: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "excluded_count": len(self.excluded_documents),
            "excluded_documents": self.excluded_documents,
            "issues": self.issues,
        }


VALID_EXCLUSION_REASONS = {reason.value for reason in ExclusionReason}


def check_exclusion_integrity(documents: list[CanonicalDocument]) -> ExclusionIntegrityReport:
    report = ExclusionIntegrityReport()

    for doc in documents:
        if not doc.is_current:
            continue

        is_excluded_status = doc.processing_status == ProcessingStatus.EXCLUDED
        is_excluded_meta = bool(doc.metadata.get("excluded"))
        reason = doc.metadata.get("exclusion_reason")

        if is_excluded_status or is_excluded_meta:
            entry = {
                "document_id": doc.document_id,
                "website": doc.website,
                "url": doc.canonical_url,
                "exclusion_reason": reason,
                "processing_status": doc.processing_status.value,
            }
            report.excluded_documents.append(entry)

            if not is_excluded_status:
                report.issues.append(
                    {
                        "document_id": doc.document_id,
                        "message": "metadata.excluded set but status is not EXCLUDED",
                    }
                )
                report.passed = False
            if not reason:
                report.issues.append(
                    {"document_id": doc.document_id, "message": "missing exclusion_reason"}
                )
                report.passed = False
            elif reason not in VALID_EXCLUSION_REASONS:
                report.issues.append(
                    {
                        "document_id": doc.document_id,
                        "message": f"unknown exclusion_reason: {reason}",
                    }
                )
                report.passed = False

        if doc.processing_status in (ProcessingStatus.ACTIVE, ProcessingStatus.VALIDATED):
            if not is_target_website(doc.website):
                report.issues.append(
                    {
                        "document_id": doc.document_id,
                        "message": "NON_TARGET document has production status",
                    }
                )
                report.passed = False
            if doc.metadata.get("excluded"):
                report.issues.append(
                    {
                        "document_id": doc.document_id,
                        "message": "excluded metadata on ACTIVE/VALIDATED document",
                    }
                )
                report.passed = False

    return report
