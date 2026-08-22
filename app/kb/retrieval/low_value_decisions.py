"""Deterministic low-value document classification for Phase 8."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from app.kb.enums import RetrievalEligibilityStatus, RetrievalExclusionReason
from app.kb.models.canonical import CanonicalDocument
from app.kb.retrieval.eligibility import assess_retrieval_eligibility
from app.kb.retrieval.models import RetrievalEligibilityDecision

LOW_VALUE_TOKEN_THRESHOLD = 100

UI_ONLY_PATTERNS = (
    re.compile(r"click the map to enable", re.I),
    re.compile(r"^send message$", re.I),
    re.compile(r"^message us$", re.I),
    re.compile(r"^drop us message", re.I),
)


class LowValueClassification(StrEnum):
    USEFUL = "USEFUL"
    NOT_USEFUL = "NOT_USEFUL"
    REVIEW = "REVIEW"


class LowValueDecisionRecord(BaseModel):
    document_id: str
    url: str
    tokens: int
    classification: LowValueClassification
    reason: str
    evidence: str


# Explicit overrides for Phase 7 low-value eligible documents.
LOW_VALUE_OVERRIDES: dict[str, tuple[LowValueClassification, str]] = {
    "3e8570ca-cb85-552d-951b-7b0311df9bf0": (
        LowValueClassification.NOT_USEFUL,
        "Map/dealer-search UI placeholder with no substantive retrieval content",
    ),
    "6a7fe029-cb09-5e9f-a9c9-4139b44775e7": (
        LowValueClassification.USEFUL,
        "Contains substantive press release headline and linked announcement",
    ),
    "db2c48ba-747f-5860-8f80-abab16bc8dfb": (
        LowValueClassification.NOT_USEFUL,
        "Mostly navigation and form UI; contact details better covered elsewhere",
    ),
    "6e036a4e-ced1-5094-97ed-29a45d442b89": (
        LowValueClassification.USEFUL,
        "Contains address, phone, and email suitable for contact queries",
    ),
    "439b5abd-dbc6-5473-902e-297b07609a81": (
        LowValueClassification.USEFUL,
        "Investor material-documents index listing IPO filing references",
    ),
    "6d773861-fb69-5eb9-b7b9-cc4e1cc32dc8": (
        LowValueClassification.REVIEW,
        "Blog pagination page; near-duplicate of blogs/news page 1",
    ),
    "de0a9f89-96f1-58b6-baac-1f562907528c": (
        LowValueClassification.USEFUL,
        "Product specification PDF with meaningful technical content",
    ),
    "8698998e-41da-5f82-b7f7-d14b1e5de278": (
        LowValueClassification.USEFUL,
        "Product specification PDF with meaningful technical content",
    ),
    "96b87bad-9cdb-58fc-b793-c4f60540b5b0": (
        LowValueClassification.USEFUL,
        "Product kit component specification PDF",
    ),
    "dcf5d8cf-a658-5231-b5ef-0fc9f6c12645": (
        LowValueClassification.USEFUL,
        "Product kit component specification PDF",
    ),
}


def approximate_tokens(text: str) -> int:
    return len(text.split())


def classify_low_value(
    canonical: CanonicalDocument,
    retrieval_text: str,
) -> LowValueDecisionRecord | None:
    tokens = approximate_tokens(retrieval_text)
    if tokens >= LOW_VALUE_TOKEN_THRESHOLD:
        return None

    override = LOW_VALUE_OVERRIDES.get(canonical.document_id)
    if override:
        classification, reason = override
        return LowValueDecisionRecord(
            document_id=canonical.document_id,
            url=canonical.canonical_url,
            tokens=tokens,
            classification=classification,
            reason=reason,
            evidence=f"tokens={tokens}, override={classification.value}",
        )

    ui_hits = sum(1 for pattern in UI_ONLY_PATTERNS if pattern.search(retrieval_text))
    if ui_hits >= 2:
        return LowValueDecisionRecord(
            document_id=canonical.document_id,
            url=canonical.canonical_url,
            tokens=tokens,
            classification=LowValueClassification.NOT_USEFUL,
            reason="Content is primarily UI placeholders",
            evidence=f"tokens={tokens}, ui_pattern_hits={ui_hits}",
        )

    return LowValueDecisionRecord(
        document_id=canonical.document_id,
        url=canonical.canonical_url,
        tokens=tokens,
        classification=LowValueClassification.REVIEW,
        reason="Short document requires manual usefulness review",
        evidence=f"tokens={tokens}",
    )


def apply_low_value_decision(
    base: RetrievalEligibilityDecision,
    record: LowValueDecisionRecord,
) -> RetrievalEligibilityDecision:
    if record.classification == LowValueClassification.NOT_USEFUL:
        return RetrievalEligibilityDecision.excluded(
            RetrievalExclusionReason.INSUFFICIENT_CONTENT,
            document_type=base.document_type,
        )
    if record.classification == LowValueClassification.REVIEW:
        from app.kb.enums import RetrievalReviewReason

        return RetrievalEligibilityDecision.review(
            RetrievalReviewReason.LOW_CONTENT_CONFIDENCE,
            document_type=base.document_type,
            notes=record.reason,
        )
    return base


def write_low_value_decisions(path: Path, records: list[LowValueDecisionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision_count": len(records),
        "decisions": [record.model_dump(mode="json") for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
