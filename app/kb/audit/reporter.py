import json
from pathlib import Path

from app.kb.audit.models import AuditSummary, DocumentAuditRecord, ReviewItem, ReviewSeverity, utc_now_iso


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(
    *,
    raw_count: int,
    cleaned_count: int,
    records: list[DocumentAuditRecord],
    duplicate_report: dict,
) -> AuditSummary:
    by_website: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    by_extraction_method: dict[str, int] = {}
    by_processing_status: dict[str, int] = {}
    by_website_source_type: dict[str, dict[str, int]] = {}

    validated = warning = failed = active = excluded = 0
    boilerplate_docs = 0
    high_loss = moderate_loss = 0
    ocr_issues = 0
    review_critical = review_warning = review_info = 0
    review_doc_ids: set[str] = set()

    for record in records:
        by_website[record.website] = by_website.get(record.website, 0) + 1
        by_source_type[record.source_type] = by_source_type.get(record.source_type, 0) + 1
        by_extraction_method[record.extraction_method] = (
            by_extraction_method.get(record.extraction_method, 0) + 1
        )
        by_processing_status[record.processing_status] = (
            by_processing_status.get(record.processing_status, 0) + 1
        )
        site_types = by_website_source_type.setdefault(
            record.website, {"html": 0, "pdf": 0, "scanned_pdf": 0}
        )
        site_types[record.source_type] = site_types.get(record.source_type, 0) + 1

        if record.processing_status == "VALIDATED":
            validated += 1
        elif record.processing_status == "FAILED":
            failed += 1
        elif record.processing_status == "ACTIVE":
            active += 1
        elif record.processing_status == "EXCLUDED":
            excluded += 1
        if record.validation.get("warnings"):
            warning += 1

        if record.boilerplate_matches:
            boilerplate_docs += 1
        if record.content_loss.get("retention_bucket") == "high_loss":
            high_loss += 1
        elif record.content_loss.get("retention_bucket") == "moderate_loss":
            moderate_loss += 1
        if record.ocr and any(
            issue
            for issue in record.ocr.get("issues", [])
            if "missing document-level OCR metadata" not in issue
        ):
            ocr_issues += 1

        for item in record.review_items:
            review_doc_ids.add(record.document_id)
            if item.severity == ReviewSeverity.CRITICAL:
                review_critical += 1
            elif item.severity == ReviewSeverity.WARNING:
                review_warning += 1
            else:
                review_info += 1

    duplicate_groups = sum(len(v) for v in duplicate_report.values())

    return AuditSummary(
        generated_at=utc_now_iso(),
        raw_count=raw_count,
        cleaned_count=cleaned_count,
        canonical_count=len(records),
        by_website=by_website,
        by_source_type=by_source_type,
        by_extraction_method=by_extraction_method,
        by_processing_status=by_processing_status,
        by_website_source_type=by_website_source_type,
        validated_count=validated,
        warning_count=warning,
        failed_count=failed,
        excluded_count=excluded,
        active_count=active,
        duplicate_groups=duplicate_groups,
        boilerplate_document_count=boilerplate_docs,
        high_content_loss_count=high_loss,
        moderate_content_loss_count=moderate_loss,
        ocr_issue_count=ocr_issues,
        review_critical=review_critical,
        review_warning=review_warning,
        review_info=review_info,
        human_review_count=len(review_doc_ids),
    )


def write_human_summary(path: Path, summary: AuditSummary, records: list[DocumentAuditRecord]) -> None:
    lines = [
        "Knowledge Base Quality Audit",
        f"Generated: {summary.generated_at}",
        "",
        "Counts",
        f"  RAW documents: {summary.raw_count}",
        f"  CLEANED documents: {summary.cleaned_count}",
        f"  CANONICAL documents: {summary.canonical_count}",
        "",
        "By website",
    ]
    for website, count in sorted(summary.by_website.items()):
        lines.append(f"  {website}: {count}")
        site_types = summary.by_website_source_type.get(website, {})
        for source_type, type_count in sorted(site_types.items()):
            if type_count:
                lines.append(f"    {source_type.upper()}: {type_count}")

    lines.extend(
        [
            "",
            "By processing status",
            f"  ACTIVE: {summary.active_count}",
            f"  VALIDATED: {summary.validated_count}",
            f"  FAILED: {summary.failed_count}",
            f"  EXCLUDED: {summary.excluded_count}",
            "",
            "Quality signals",
            f"  Duplicate groups: {summary.duplicate_groups}",
            f"  Documents with residual boilerplate: {summary.boilerplate_document_count}",
            f"  High content loss (<50% baseline): {summary.high_content_loss_count}",
            f"  Moderate content loss (50-70%): {summary.moderate_content_loss_count}",
            f"  OCR issues: {summary.ocr_issue_count}",
            "",
            "Human review queue",
            f"  Documents requiring review: {summary.human_review_count}",
            f"  CRITICAL items: {summary.review_critical}",
            f"  WARNING items: {summary.review_warning}",
            f"  INFO items: {summary.review_info}",
            "",
        ]
    )

    critical_docs = [
        r for r in records if any(i.severity == ReviewSeverity.CRITICAL for i in r.review_items)
    ]
    if critical_docs:
        lines.append("Critical documents (sample):")
        for record in critical_docs[:20]:
            issues = [i.issue for i in record.review_items if i.severity == ReviewSeverity.CRITICAL]
            lines.append(f"  - {record.website} | {record.title} | {record.canonical_url}")
            for issue in issues[:3]:
                lines.append(f"      {issue}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    reports_dir: Path,
    summary: AuditSummary,
    records: list[DocumentAuditRecord],
    duplicate_report: dict,
    review_items: list[ReviewItem],
) -> dict[str, Path]:
    paths = {
        "summary_json": reports_dir / "kb_audit_summary.json",
        "documents_json": reports_dir / "kb_audit_documents.json",
        "duplicates_json": reports_dir / "kb_audit_duplicates.json",
        "review_json": reports_dir / "kb_audit_review.json",
        "summary_txt": reports_dir / "kb_audit_summary.txt",
    }

    write_json(paths["summary_json"], summary.to_dict())
    write_json(paths["documents_json"], [record.to_dict() for record in records])
    write_json(paths["duplicates_json"], duplicate_report)
    write_json(
        paths["review_json"],
        [
            {
                "document_id": item.document_id,
                "website": item.website,
                "url": item.url,
                "title": item.title,
                "source_type": item.source_type,
                "issue": item.issue,
                "severity": item.severity,
                "context": item.context,
            }
            for item in review_items
        ],
    )
    write_human_summary(paths["summary_txt"], summary, records)
    return paths
