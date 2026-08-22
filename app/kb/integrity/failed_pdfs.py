"""Analyze FAILED investor PDFs for recoverability."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.kb.audit.collect import DocumentBundle

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_READABLE_RE = re.compile(r"[\w\s.,;:!?()\-/'\"]", re.UNICODE)


@dataclass
class FailedPdfAssessment:
    document_id: str
    website: str
    url: str
    title: str
    classification: str  # RECOVERABLE | UNRECOVERABLE
    recommended_action: str
    metrics: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "website": self.website,
            "url": self.url,
            "title": self.title,
            "classification": self.classification,
            "recommended_action": self.recommended_action,
            "metrics": self.metrics,
            "notes": self.notes,
        }


def _readable_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    readable = len(_READABLE_RE.findall(text))
    return readable / max(len(text), 1)


def _control_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CONTROL_CHAR_RE.findall(text)) / max(len(text), 1)


def assess_failed_pdf(bundle: DocumentBundle) -> FailedPdfAssessment:
    canonical = bundle.canonical
    raw = bundle.raw
    cleaned = bundle.cleaned

    raw_text = raw.content if raw else ""
    cleaned_text = cleaned.content if cleaned else ""
    plain = canonical.plain_text or ""

    raw_readable = _readable_char_ratio(raw_text)
    cleaned_readable = _readable_char_ratio(cleaned_text)
    control_ratio = _control_char_ratio(cleaned_text or plain)

    notes: list[str] = []
    classification = "UNRECOVERABLE"
    action = "Mark EXCLUDED with reason extraction_failed"

    if control_ratio > 0.3:
        notes.append(f"High control-character ratio ({control_ratio:.0%}) indicates custom font encoding failure")
    if cleaned_readable < 0.2:
        notes.append(f"Cleaned text is not human-readable ({cleaned_readable:.0%} readable chars)")
    if raw_readable < 0.25:
        notes.append(f"RAW native PDF extraction is garbled ({raw_readable:.0%} readable chars)")

    if canonical.extraction_method.value == "pdf_text" and control_ratio > 0.3:
        notes.append(
            "Existing OCR fallback is available in pdf_extractor but requires re-ingesting source PDF; "
            "not applied during freeze phase"
        )

    # Recoverable only if meaningful readable text exists without control-char corruption
    if cleaned_readable >= 0.5 and control_ratio < 0.05:
        classification = "RECOVERABLE"
        action = "Reprocess with existing pipeline — content appears intact"

    return FailedPdfAssessment(
        document_id=canonical.document_id,
        website=canonical.website,
        url=canonical.canonical_url,
        title=canonical.title,
        classification=classification,
        recommended_action=action,
        metrics={
            "raw_char_count": len(raw_text),
            "cleaned_char_count": len(cleaned_text),
            "plain_text_char_count": len(plain),
            "raw_readable_ratio": round(raw_readable, 4),
            "cleaned_readable_ratio": round(cleaned_readable, 4),
            "control_char_ratio": round(control_ratio, 4),
            "extraction_method": canonical.extraction_method.value,
        },
        notes=notes,
    )


def assess_all_failed_pdfs(bundles: list[DocumentBundle]) -> list[FailedPdfAssessment]:
    assessments: list[FailedPdfAssessment] = []
    for bundle in bundles:
        if bundle.canonical.processing_status.value != "FAILED":
            continue
        assessments.append(assess_failed_pdf(bundle))
    return assessments
