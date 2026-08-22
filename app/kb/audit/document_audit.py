import re
from urllib.parse import urlparse

from app.kb.audit.analysis import audit_ocr, audit_structure, compute_content_loss
from app.kb.audit.collect import DocumentBundle
from app.kb.audit.duplicates import scan_boilerplate
from app.kb.audit.models import DocumentAuditRecord, ReviewItem, ReviewSeverity
from app.kb.enums import ProcessingStatus, SourceType
from app.kb.validation.document_validator import MIN_CLEANED_MEANINGFUL_CHARS, _meaningful_char_count

INVALID_URL_SCHEMES = frozenset({"", "mailto", "tel", "javascript"})


def _is_valid_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def audit_document(bundle: DocumentBundle) -> DocumentAuditRecord:
    canonical = bundle.canonical
    raw = bundle.raw
    cleaned = bundle.cleaned

    identity_issues: list[str] = []
    review_items: list[ReviewItem] = []

    def add_review(issue: str, severity: ReviewSeverity, context: dict | None = None) -> None:
        review_items.append(
            ReviewItem(
                document_id=canonical.document_id,
                website=canonical.website,
                url=canonical.canonical_url,
                title=canonical.title,
                source_type=canonical.source_type.value,
                issue=issue,
                severity=severity,
                context=context or {},
            )
        )

    if not canonical.document_id:
        identity_issues.append("missing document_id")
        add_review("missing document_id", ReviewSeverity.CRITICAL)
    if not canonical.canonical_url:
        identity_issues.append("missing canonical_url")
        add_review("missing canonical_url", ReviewSeverity.CRITICAL)
    if not canonical.source_url:
        identity_issues.append("missing source_url")
        add_review("missing source_url", ReviewSeverity.CRITICAL)
    elif not _is_valid_url(canonical.source_url):
        identity_issues.append("invalid source_url")
        add_review("invalid source_url", ReviewSeverity.CRITICAL, {"url": canonical.source_url})

    if not canonical.content_hash or not canonical.content_hash.startswith("sha256:"):
        identity_issues.append("invalid content_hash")
        add_review("invalid content_hash", ReviewSeverity.CRITICAL)

    plain_text = canonical.plain_text or ""
    cleaned_text = cleaned.content if cleaned else ""
    if not cleaned_text.strip() and not plain_text.strip():
        add_review("empty document content", ReviewSeverity.CRITICAL)

    if not canonical.title.strip():
        add_review("missing title", ReviewSeverity.WARNING)

    cleaned_meaningful = _meaningful_char_count(cleaned_text)
    if canonical.source_type == SourceType.HTML and cleaned_meaningful < MIN_CLEANED_MEANINGFUL_CHARS:
        add_review(
            f"extremely short content ({cleaned_meaningful} meaningful chars)",
            ReviewSeverity.WARNING,
        )

    validation_meta = canonical.metadata.get("validation", {})
    validation_errors = validation_meta.get("errors", [])
    validation_warnings = validation_meta.get("warnings", [])

    for error in validation_errors:
        add_review(f"validation failed: {error}", ReviewSeverity.CRITICAL)
    for warning in validation_warnings:
        add_review(f"validation warning: {warning}", ReviewSeverity.WARNING)

    if canonical.processing_status == ProcessingStatus.FAILED:
        add_review("processing status FAILED", ReviewSeverity.CRITICAL)
    elif canonical.processing_status == ProcessingStatus.EXCLUDED:
        add_review(
            f"excluded from production KB: {canonical.metadata.get('exclusion_reason', 'unknown')}",
            ReviewSeverity.INFO,
        )

    content_loss = compute_content_loss(bundle)
    if content_loss["retention_bucket"] == "high_loss":
        add_review(
            "high content reduction vs baseline",
            ReviewSeverity.WARNING,
            {
                "retention_vs_baseline": content_loss["retention_vs_baseline"],
                "retention_vs_raw": content_loss["retention_vs_raw"],
            },
        )
    elif content_loss["retention_bucket"] == "moderate_loss":
        add_review(
            "moderate content reduction vs baseline",
            ReviewSeverity.WARNING,
            {"retention_vs_baseline": content_loss["retention_vs_baseline"]},
        )

    structure = audit_structure(bundle)
    for issue in structure.get("issues", []):
        severity = ReviewSeverity.WARNING
        if "out of order" in issue or "missing pdf pages" in issue:
            severity = ReviewSeverity.CRITICAL
        add_review(f"structure issue: {issue}", severity)

    ocr_report = audit_ocr(bundle)
    if ocr_report:
        for issue in ocr_report.get("issues", []):
            severity = ReviewSeverity.WARNING
            if "missing document-level OCR metadata" in issue:
                severity = ReviewSeverity.INFO
            add_review(f"ocr issue: {issue}", severity)

    boilerplate_matches = scan_boilerplate(bundle)
    for match in boilerplate_matches:
        add_review(
            f"residual boilerplate ({match.category})",
            match.severity,
            {"pattern": match.pattern, "context": match.context},
        )

    internal_links = sum(1 for link in canonical.links if link.internal)
    external_links = sum(1 for link in canonical.links if not link.internal)
    if external_links > 0 and internal_links == 0 and cleaned_meaningful < 200:
        add_review("document has external links only", ReviewSeverity.INFO, {"external_links": external_links})

    pdf_info = None
    if canonical.source_type in (SourceType.PDF, SourceType.SCANNED_PDF):
        pages = canonical.structured_content.pages or []
        pdf_info = {
            "page_count": len(pages),
            "page_numbers": [p.page_number for p in pages],
            "ordered": [p.page_number for p in pages] == sorted(p.page_number for p in pages),
        }

    return DocumentAuditRecord(
        document_id=canonical.document_id,
        website=canonical.website,
        canonical_url=canonical.canonical_url,
        source_url=canonical.source_url,
        title=canonical.title,
        source_type=canonical.source_type.value,
        extraction_method=canonical.extraction_method.value,
        processing_status=canonical.processing_status.value,
        content_hash=canonical.content_hash,
        raw_content_hash=canonical.metadata.get("raw_content_hash"),
        cleaned_content_hash=canonical.metadata.get("cleaned_content_hash"),
        identity={"issues": identity_issues},
        content={
            "title": canonical.title,
            "plain_text_length": len(plain_text),
            "cleaned_length": len(cleaned_text),
            "has_structured_content": bool(canonical.structured_content),
        },
        pdf=pdf_info,
        ocr=ocr_report,
        links={
            "total": len(canonical.links),
            "internal": internal_links,
            "external": external_links,
        },
        content_loss=content_loss,
        structure=structure,
        boilerplate_matches=boilerplate_matches,
        review_items=review_items,
        validation={
            "errors": validation_errors,
            "warnings": validation_warnings,
            "status": canonical.processing_status.value,
        },
    )
