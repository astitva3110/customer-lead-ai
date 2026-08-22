"""Deterministic Phase 8 review-queue decisions for retrieval eligibility."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from app.kb.enums import (
    DocumentType,
    ExtractionMethod,
    RetrievalEligibilityStatus,
    RetrievalExclusionReason,
    RetrievalReviewReason,
)
from app.kb.models.canonical import CanonicalDocument
from app.kb.retrieval.eligibility import OCR_GARBAGE_PATTERN, assess_retrieval_eligibility
from app.kb.retrieval.models import RetrievalEligibilityDecision

OCR_GARBAGE_HEAVY = re.compile(r"(.)\1{4,}|[^a-zA-Z0-9\s.,;:!?()\-–—/₹$%@&'\"]{3,}")

IN_SCOPE_KEYWORDS = re.compile(
    r"\b("
    r"hearing aid|hearing care|earkart|radius|eqfy|tiny|fame|bluup|omni|"
    r"warranty|return|replacement|refund|shipping|specification|"
    r"investor|annual report|prospectus|policy|faq|"
    r"battery|decibel|db spl|channels|program|"
    r"company|director|shareholder|agm|board meeting|"
    r"pm cares|csr|receipt|contribution|patent|posh|insider"
    r")\b",
    re.I,
)

AMBIGUOUS_REVIEW_URLS = frozenset(
    {
        "https://earkart.in/tlm/tlm1.pdf",
        "https://earkart.in/tlm/tlm2.pdf",
        "https://earkart.in/tlm/tlm3.pdf",
    }
)


class ReviewDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    KEEP_REVIEW = "KEEP_REVIEW"


class ReviewDecisionRecord(BaseModel):
    document_id: str
    url: str
    current_status: str
    decision: ReviewDecision
    reason: str
    evidence: str
    confidence: str = Field(description="high | medium | low")


def grade_ocr_text(text: str) -> str:
    """Deterministic OCR quality grade aligned with Phase 7 audit thresholds."""
    if not text.strip():
        return "UNUSABLE"
    garbage_signals = len(OCR_GARBAGE_HEAVY.findall(text))
    if garbage_signals >= 6:
        return "UNUSABLE"
    if garbage_signals >= 4:
        return "SUBSTANTIAL"
    if garbage_signals >= 1:
        return "MINOR"
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(text) > 400 and len(words) < 25:
        return "UNUSABLE"
    return "GOOD"


def decide_review_document(canonical: CanonicalDocument, base: RetrievalEligibilityDecision) -> ReviewDecisionRecord:
    """Apply deterministic evidence rules to one REVIEW document."""
    url = canonical.canonical_url
    text = canonical.plain_text or ""
    meaningful_chars = len(re.sub(r"\s+", "", text))
    doc_type = base.document_type

    if url.rstrip("/") in {u.rstrip("/") for u in AMBIGUOUS_REVIEW_URLS}:
        return ReviewDecisionRecord(
            document_id=canonical.document_id,
            url=url,
            current_status="review",
            decision=ReviewDecision.KEEP_REVIEW,
            reason="Document type and production RAG relevance remain uncertain",
            evidence=f"type={doc_type.value}, meaningful_chars={meaningful_chars}, url matches tlm/* pattern",
            confidence="high",
        )

    if canonical.extraction_method == ExtractionMethod.OCR:
        grade = grade_ocr_text(text)
        if grade == "UNUSABLE":
            return ReviewDecisionRecord(
                document_id=canonical.document_id,
                url=url,
                current_status="review",
                decision=ReviewDecision.KEEP_REVIEW,
                reason="OCR grade UNUSABLE — content not reliably readable for retrieval",
                evidence=f"ocr_grade={grade}, garbage_signals={len(OCR_GARBAGE_HEAVY.findall(text))}, meaningful_chars={meaningful_chars}",
                confidence="high",
            )
        if grade == "SUBSTANTIAL":
            return ReviewDecisionRecord(
                document_id=canonical.document_id,
                url=url,
                current_status="review",
                decision=ReviewDecision.KEEP_REVIEW,
                reason="OCR grade SUBSTANTIAL — errors may materially affect retrieval accuracy",
                evidence=f"ocr_grade={grade}, garbage_signals={len(OCR_GARBAGE_HEAVY.findall(text))}",
                confidence="high",
            )

    in_scope = doc_type in {
        DocumentType.INVESTOR,
        DocumentType.NOTICE,
        DocumentType.POLICY,
        DocumentType.PRODUCT,
    } or bool(IN_SCOPE_KEYWORDS.search(text))

    if doc_type == DocumentType.OTHER and not in_scope and meaningful_chars < 500:
        return ReviewDecisionRecord(
            document_id=canonical.document_id,
            url=url,
            current_status="review",
            decision=ReviewDecision.KEEP_REVIEW,
            reason="Ambiguous relevance with insufficient in-scope signals",
            evidence=f"type={doc_type.value}, meaningful_chars={meaningful_chars}, in_scope_keywords={bool(IN_SCOPE_KEYWORDS.search(text))}",
            confidence="medium",
        )

    if meaningful_chars >= 200 and in_scope:
        grade = grade_ocr_text(text) if canonical.extraction_method == ExtractionMethod.OCR else "GOOD"
        return ReviewDecisionRecord(
            document_id=canonical.document_id,
            url=url,
            current_status="review",
            decision=ReviewDecision.APPROVE,
            reason="Content is meaningful, in-scope, and sufficiently readable for retrieval",
            evidence=f"type={doc_type.value}, ocr_grade={grade}, meaningful_chars={meaningful_chars}, review_reason={base.review_reason.value if base.review_reason else None}",
            confidence="high" if grade == "GOOD" else "medium",
        )

    return ReviewDecisionRecord(
        document_id=canonical.document_id,
        url=url,
        current_status="review",
        decision=ReviewDecision.KEEP_REVIEW,
        reason="Insufficient evidence to approve or reject automatically",
        evidence=f"type={doc_type.value}, meaningful_chars={meaningful_chars}, in_scope={in_scope}",
        confidence="low",
    )


def apply_review_decision(
    base: RetrievalEligibilityDecision,
    record: ReviewDecisionRecord,
) -> RetrievalEligibilityDecision:
    """Override base eligibility using an explicit review decision."""
    if record.decision == ReviewDecision.APPROVE:
        return RetrievalEligibilityDecision.eligible(document_type=base.document_type)
    if record.decision == ReviewDecision.REJECT:
        return RetrievalEligibilityDecision.excluded(
            RetrievalExclusionReason.OUT_OF_SCOPE,
            document_type=base.document_type,
        )
    return base


def build_review_decisions(canonicals: list[CanonicalDocument]) -> list[ReviewDecisionRecord]:
    """Evaluate all REVIEW documents deterministically."""
    records: list[ReviewDecisionRecord] = []
    for canonical in canonicals:
        base = assess_retrieval_eligibility(canonical)
        if base.status != RetrievalEligibilityStatus.REVIEW:
            continue
        records.append(decide_review_document(canonical, base))
    records.sort(key=lambda item: item.url)
    return records


def load_review_decisions(path: Path) -> dict[str, ReviewDecisionRecord]:
    """Load persisted review decisions keyed by document_id."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [ReviewDecisionRecord.model_validate(item) for item in payload.get("decisions", payload)]
    return {record.document_id: record for record in records}


def write_review_decisions(path: Path, records: list[ReviewDecisionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "decision_count": len(records),
        "approve_count": sum(1 for r in records if r.decision == ReviewDecision.APPROVE),
        "reject_count": sum(1 for r in records if r.decision == ReviewDecision.REJECT),
        "keep_review_count": sum(1 for r in records if r.decision == ReviewDecision.KEEP_REVIEW),
        "decisions": [record.model_dump(mode="json") for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
