import re

from app.kb.audit.collect import DocumentBundle
from app.kb.enums import SourceType
from app.kb.validation.document_validator import _meaningful_char_count, baseline_text

MIN_EXTREMELY_SHORT_ALNUM = 40
MIN_SHORT_PAGE_CHARS = 50
LOW_OCR_CONFIDENCE = 50.0
GIANT_PARAGRAPH_CHARS = 5000
TINY_PARAGRAPH_THRESHOLD = 500


def compute_content_loss(bundle: DocumentBundle) -> dict:
    raw = bundle.raw
    cleaned = bundle.cleaned
    canonical = bundle.canonical

    raw_chars = len(raw.content) if raw else 0
    cleaned_chars = len(cleaned.content) if cleaned else 0
    canonical_chars = len(canonical.plain_text or "")

    raw_alnum = _meaningful_char_count(raw.content) if raw else 0
    baseline = baseline_text(raw) if raw else ""
    baseline_alnum = _meaningful_char_count(baseline) if raw else 0
    cleaned_alnum = _meaningful_char_count(cleaned.content) if cleaned else 0

    retention_vs_raw = (cleaned_alnum / raw_alnum) if raw_alnum else None
    retention_vs_baseline = (cleaned_alnum / baseline_alnum) if baseline_alnum else None

    bucket = "normal"
    if retention_vs_baseline is not None:
        if retention_vs_baseline < 0.50:
            bucket = "high_loss"
        elif retention_vs_baseline < 0.70:
            bucket = "moderate_loss"
    if retention_vs_raw is not None and retention_vs_raw > 0.98 and raw_chars > 5000:
        bucket = "low_reduction"

    return {
        "raw_char_count": raw_chars,
        "cleaned_char_count": cleaned_chars,
        "canonical_plain_text_char_count": canonical_chars,
        "raw_alnum_count": raw_alnum,
        "baseline_alnum_count": baseline_alnum,
        "cleaned_alnum_count": cleaned_alnum,
        "retention_vs_raw": round(retention_vs_raw, 4) if retention_vs_raw is not None else None,
        "retention_vs_baseline": round(retention_vs_baseline, 4) if retention_vs_baseline is not None else None,
        "retention_bucket": bucket,
    }


def _walk_nodes(nodes, stats: dict, issues: list[str]) -> None:
    for node in nodes:
        node_type = getattr(node, "type", None)
        if node_type == "paragraph":
            stats["paragraphs"] += 1
            text = node.text.strip()
            if not text:
                issues.append("empty paragraph")
            elif len(text) > GIANT_PARAGRAPH_CHARS:
                issues.append(f"giant paragraph ({len(text)} chars)")
            elif len(text) < 20:
                stats["tiny_paragraphs"] += 1
        elif node_type == "heading":
            stats["headings"] += 1
            if node.level > 4:
                stats["deep_headings"] += 1
        elif node_type == "list":
            stats["lists"] += 1
            if not node.items:
                issues.append("empty list")
        elif node_type == "table":
            stats["tables"] += 1
            if not node.headers and not node.rows:
                issues.append("empty table")
            elif node.headers and not node.rows:
                issues.append("table with headers but no rows")
        elif node_type == "section":
            stats["sections"] += 1
            if not node.heading and not node.children:
                issues.append("empty section")
            elif node.heading:
                key = node.heading.lower()
                stats["section_headings"][key] = stats["section_headings"].get(key, 0) + 1
            _walk_nodes(node.children, stats, issues)


def audit_structure(bundle: DocumentBundle) -> dict:
    canonical = bundle.canonical
    structured = canonical.structured_content
    stats = {
        "sections": 0,
        "headings": 0,
        "paragraphs": 0,
        "lists": 0,
        "tables": 0,
        "deep_headings": 0,
        "tiny_paragraphs": 0,
        "section_headings": {},
    }
    issues: list[str] = []

    if not structured:
        return {"valid": False, "issues": ["missing structured_content root"], **stats}

    if structured.pages:
        stats["pdf_page_count"] = len(structured.pages)
        page_numbers = [p.page_number for p in structured.pages]
        if page_numbers != sorted(page_numbers):
            issues.append("pdf pages out of order")
        if len(set(page_numbers)) != len(page_numbers):
            issues.append("duplicate pdf page numbers")
        expected = set(range(1, max(page_numbers) + 1)) if page_numbers else set()
        actual = set(page_numbers)
        missing = sorted(expected - actual)
        if missing:
            issues.append(f"missing pdf pages: {missing[:10]}")
        for page in structured.pages:
            if not page.blocks:
                issues.append(f"empty pdf page {page.page_number}")
            _walk_nodes(page.blocks, stats, issues)
    else:
        if not structured.children:
            issues.append("document has no structured children")
        _walk_nodes(structured.children, stats, issues)

    duplicated_sections = [h for h, c in stats["section_headings"].items() if c > 1]
    if duplicated_sections:
        issues.append(f"duplicated section headings: {duplicated_sections[:5]}")

    if stats["paragraphs"] == 1 and canonical.plain_text and len(canonical.plain_text) > 2000:
        issues.append("single giant paragraph structure")

    if stats["tiny_paragraphs"] > TINY_PARAGRAPH_THRESHOLD:
        issues.append(f"excessive tiny paragraphs ({stats['tiny_paragraphs']})")

    return {
        "valid": not issues,
        "issues": issues,
        "sections": stats["sections"],
        "headings": stats["headings"],
        "paragraphs": stats["paragraphs"],
        "lists": stats["lists"],
        "tables": stats["tables"],
        "pdf_page_count": stats.get("pdf_page_count"),
    }


def audit_ocr(bundle: DocumentBundle) -> dict | None:
    canonical = bundle.canonical
    if canonical.source_type != SourceType.SCANNED_PDF and canonical.extraction_method.value != "ocr":
        return None

    structured = canonical.structured_content
    ocr_meta = structured.ocr or (bundle.raw.ocr.model_dump() if bundle.raw and bundle.raw.ocr else None)
    if ocr_meta is not None and hasattr(ocr_meta, "model_dump"):
        ocr_meta = ocr_meta.model_dump()

    page_reports: list[dict] = []
    issues: list[str] = []
    suspicious_re = re.compile(r"(\|\s*){3,}|\.{4,}")

    if not ocr_meta:
        issues.append("missing document-level OCR metadata (confidence may be unavailable from extractor)")

    pages = structured.pages or []
    for page in pages:
        page_text = "\n".join(
            block.text for block in page.blocks if hasattr(block, "text") and block.text
        )
        char_count = len(page_text)
        confidence = page.ocr_confidence
        page_issues: list[str] = []

        if char_count < MIN_SHORT_PAGE_CHARS:
            page_issues.append("unusually short page")
        if confidence is not None and confidence < LOW_OCR_CONFIDENCE:
            page_issues.append(f"low OCR confidence ({confidence:.1f})")
        elif confidence is None and ocr_meta is None:
            pass  # confidence genuinely unavailable — not flagged per-page
        if suspicious_re.search(page_text):
            page_issues.append("suspicious OCR characters")

        if page_issues:
            issues.extend([f"page {page.page_number}: {item}" for item in page_issues])

        page_reports.append(
            {
                "page_number": page.page_number,
                "char_count": char_count,
                "ocr_confidence": confidence,
                "issues": page_issues,
            }
        )

    return {
        "extraction_method": canonical.extraction_method.value,
        "ocr_metadata": ocr_meta,
        "page_count": len(pages),
        "pages": page_reports,
        "issues": issues,
        "severity_hint": "WARNING" if issues else "INFO",
    }
