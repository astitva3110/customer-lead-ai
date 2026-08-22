"""Deterministic retrieval eligibility gate."""

from __future__ import annotations

import re

from app.kb.enums import (
    DocumentType,
    ExtractionMethod,
    RetrievalEligibilityStatus,
    RetrievalExclusionReason,
    RetrievalReviewReason,
    SourceType,
)
from app.kb.models.canonical import CanonicalDocument
from app.kb.retrieval.classifier import classify_document
from app.kb.retrieval.models import RetrievalEligibilityDecision
from app.kb.validation.document_validator import _meaningful_char_count

RETRIEVAL_EXCLUDED_URL_PATTERNS: tuple[tuple[str, RetrievalExclusionReason], ...] = (
    ("/account/login", RetrievalExclusionReason.ACCOUNT_UI),
    ("/account/register", RetrievalExclusionReason.ACCOUNT_UI),
    ("/account/", RetrievalExclusionReason.ACCOUNT_UI),
    ("/cart", RetrievalExclusionReason.CART_CHECKOUT_UI),
    ("/checkout", RetrievalExclusionReason.CART_CHECKOUT_UI),
    ("/apps/", RetrievalExclusionReason.THIRD_PARTY_APP_UI),
    ("/search", RetrievalExclusionReason.SEARCH_UI),
    ("/collections/", RetrievalExclusionReason.COLLECTION_CATALOG_UI),
    ("/password", RetrievalExclusionReason.ACCOUNT_UI),
    ("/challenge", RetrievalExclusionReason.TRANSACTIONAL_UI),
)

RETRIEVAL_EXCLUDED_TITLE_PATTERNS: tuple[tuple[str, RetrievalExclusionReason], ...] = (
    ("sign in", RetrievalExclusionReason.ACCOUNT_UI),
    ("log in", RetrievalExclusionReason.ACCOUNT_UI),
    ("create account", RetrievalExclusionReason.ACCOUNT_UI),
    ("your shopping cart", RetrievalExclusionReason.CART_CHECKOUT_UI),
    ("your cart", RetrievalExclusionReason.CART_CHECKOUT_UI),
    ("404 not found", RetrievalExclusionReason.NOT_FOUND_PAGE),
    ("page not found", RetrievalExclusionReason.NOT_FOUND_PAGE),
)

UI_CONTENT_PATTERNS: tuple[tuple[re.Pattern[str], RetrievalExclusionReason], ...] = (
    (re.compile(r"sign in or create an account", re.I), RetrievalExclusionReason.ACCOUNT_UI),
    (re.compile(r"email me with news and offers", re.I), RetrievalExclusionReason.ACCOUNT_UI),
    (re.compile(r"your cart is empty", re.I), RetrievalExclusionReason.CART_CHECKOUT_UI),
    (re.compile(r"taxes and shipping calculated at checkout", re.I), RetrievalExclusionReason.CART_CHECKOUT_UI),
    (re.compile(r"filter and sort", re.I), RetrievalExclusionReason.COLLECTION_CATALOG_UI),
    (re.compile(r"track your order", re.I), RetrievalExclusionReason.TRACKING_APP_UI),
    (re.compile(r"17track", re.I), RetrievalExclusionReason.TRACKING_APP_UI),
)

IN_SCOPE_KEYWORDS = re.compile(
    r"\b("
    r"hearing aid|hearing care|earkart|radius|eqfy|tiny|fame|bluup|"
    r"warranty|return|replacement|refund|shipping|specification|"
    r"investor|annual report|prospectus|policy|faq|"
    r"battery|decibel|db spl|channels|program|"
    r"company|director|shareholder|agm|board meeting|"
    r"pm cares|csr|receipt|contribution"
    r")\b",
    re.I,
)

OCR_GARBAGE_PATTERN = re.compile(r"(.)\1{4,}|[^a-zA-Z0-9\s.,;:!?()\-–—/₹$%@&'\"]{3,}")
MEANINGFUL_RECEIPT_PATTERN = re.compile(
    r"\b(receipt|inr|rs\.?|contribution|thank you|earkart|pm cares|fund)\b",
    re.I,
)
CSR_RECEIPT_PATTERN = re.compile(
    r"\b(pm cares|receipt no|received with thanks|contribution to)\b",
    re.I,
)


def _collect_text(canonical: CanonicalDocument) -> str:
    return canonical.plain_text or ""


def _url_exclusion(url: str) -> RetrievalExclusionReason | None:
    lowered = url.lower()
    for pattern, reason in RETRIEVAL_EXCLUDED_URL_PATTERNS:
        if pattern in lowered:
            return reason
    return None


def _title_exclusion(title: str) -> RetrievalExclusionReason | None:
    lowered = title.lower()
    for pattern, reason in RETRIEVAL_EXCLUDED_TITLE_PATTERNS:
        if pattern in lowered:
            return reason
    return None


def _content_ui_exclusion(text: str) -> RetrievalExclusionReason | None:
    for pattern, reason in UI_CONTENT_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def _navigation_only(text: str, meaningful_chars: int) -> bool:
    if meaningful_chars >= 300:
        return False
    link_markers = text.count("](http")
    return link_markers >= 5 and meaningful_chars < 200


def _ocr_quality_concern(text: str) -> bool:
    if not text.strip():
        return False
    garbage_hits = len(OCR_GARBAGE_PATTERN.findall(text))
    if garbage_hits >= 2:
        return True
    words = re.findall(r"[A-Za-z]{2,}", text)
    if not words:
        return True
    short_word_ratio = sum(1 for word in words if len(word) <= 2) / len(words)
    if short_word_ratio > 0.25:
        return True
    noisy_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        alpha_ratio = sum(char.isalpha() for char in stripped) / len(stripped)
        if alpha_ratio < 0.45:
            noisy_lines += 1
    return noisy_lines >= 3


def _ocr_substantially_corrupted(text: str) -> bool:
    """Detect OCR corruption severe enough to warrant human review before indexing."""
    if not text.strip():
        return False
    garbage_hits = len(OCR_GARBAGE_PATTERN.findall(text))
    if garbage_hits >= 3:
        return True
    severely_noisy_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        alpha_ratio = sum(char.isalpha() for char in stripped) / len(stripped)
        if alpha_ratio < 0.35:
            severely_noisy_lines += 1
    if severely_noisy_lines >= 5:
        return True
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(text) > 400 and len(words) < 25:
        return True
    return False


def _has_in_scope_signals(text: str, document_type: DocumentType, url: str) -> bool:
    if document_type in {
        DocumentType.PRODUCT,
        DocumentType.POLICY,
        DocumentType.FAQ,
        DocumentType.INVESTOR,
        DocumentType.NOTICE,
        DocumentType.BLOG,
    }:
        return True
    if IN_SCOPE_KEYWORDS.search(text):
        return True
    if "/investor/" in url.lower() or "/radius/" in url.lower() or "/products/" in url.lower():
        return True
    return False


def assess_retrieval_eligibility(canonical: CanonicalDocument) -> RetrievalEligibilityDecision:
    """Determine retrieval eligibility without modifying canonical content."""
    url = canonical.canonical_url
    title = canonical.title
    text = _collect_text(canonical)
    meaningful_chars = _meaningful_char_count(text)
    document_type = classify_document(
        canonical_url=url,
        title=title,
        source_type=canonical.source_type,
        structured_content=canonical.structured_content,
    )

    if reason := _url_exclusion(url):
        return RetrievalEligibilityDecision.excluded(reason, document_type=document_type)

    if reason := _title_exclusion(title):
        return RetrievalEligibilityDecision.excluded(reason, document_type=document_type)

    if reason := _content_ui_exclusion(text):
        return RetrievalEligibilityDecision.excluded(reason, document_type=document_type)

    if meaningful_chars < 80:
        return RetrievalEligibilityDecision.excluded(
            RetrievalExclusionReason.INSUFFICIENT_CONTENT,
            document_type=document_type,
        )

    if _navigation_only(text, meaningful_chars):
        return RetrievalEligibilityDecision.excluded(
            RetrievalExclusionReason.NAVIGATION_ONLY,
            document_type=document_type,
        )

    in_scope = _has_in_scope_signals(text, document_type, url)

    if canonical.extraction_method == ExtractionMethod.OCR and _ocr_substantially_corrupted(text):
        if in_scope or MEANINGFUL_RECEIPT_PATTERN.search(text):
            return RetrievalEligibilityDecision.review(
                RetrievalReviewReason.OCR_QUALITY_CONCERN,
                document_type=document_type,
                notes="OCR appears substantially corrupted but document may contain useful knowledge",
            )
        return RetrievalEligibilityDecision.review(
            RetrievalReviewReason.OCR_QUALITY_CONCERN,
            document_type=document_type,
        )

    if (
        canonical.extraction_method == ExtractionMethod.OCR
        and _ocr_quality_concern(text)
        and CSR_RECEIPT_PATTERN.search(text)
    ):
        return RetrievalEligibilityDecision.review(
            RetrievalReviewReason.OCR_QUALITY_CONCERN,
            document_type=document_type,
            notes="CSR/receipt OCR contains noise but may contain useful company contribution knowledge",
        )

    if document_type == DocumentType.OTHER and not in_scope:
        if meaningful_chars < 250:
            return RetrievalEligibilityDecision.review(
                RetrievalReviewReason.AMBIGUOUS_RELEVANCE,
                document_type=document_type,
            )
        return RetrievalEligibilityDecision.review(
            RetrievalReviewReason.AMBIGUOUS_RELEVANCE,
            document_type=document_type,
            notes="Document type uncertain; relevance to production RAG scope unclear",
        )

    if not in_scope:
        return RetrievalEligibilityDecision.excluded(
            RetrievalExclusionReason.OUT_OF_SCOPE,
            document_type=document_type,
        )

    return RetrievalEligibilityDecision.eligible(document_type=document_type)
