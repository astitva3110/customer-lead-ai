import re
from urllib.parse import urlparse

from app.kb.audit.models import BoilerplateMatch, ReviewItem, ReviewSeverity
from app.kb.audit.collect import DocumentBundle
from app.kb.url_normalizer import INDEX_ALIASES, normalize_url

EARKART_IN_PATTERNS: list[tuple[str, str, ReviewSeverity]] = [
    (r"###\s+Quick Links", "navigation", ReviewSeverity.WARNING),
    (r"##\s+Quick Links", "navigation", ReviewSeverity.WARNING),
    (r"Become Our Partner", "partner_cta", ReviewSeverity.WARNING),
    (r"###\s+Contact Information", "footer", ReviewSeverity.WARNING),
    (r"##\s+Contact Information", "footer", ReviewSeverity.WARNING),
    (r"Make An Appointment", "appointment_widget", ReviewSeverity.WARNING),
    (r"Book Appointment", "appointment_widget", ReviewSeverity.WARNING),
    (r"help-desk|helpdesk|branding-resources\.s3\.", "helpdesk", ReviewSeverity.WARNING),
    (r"Copyright ©", "footer", ReviewSeverity.INFO),
    (r"CIN:\s*L74999", "footer", ReviewSeverity.INFO),
    (r"Call Today", "footer", ReviewSeverity.INFO),
]

EARKART_COM_PATTERNS: list[tuple[str, str, ReviewSeverity]] = [
    (r"Your cart is empty", "shopify_cart", ReviewSeverity.WARNING),
    (r"##\s+Subtotal", "shopify_cart", ReviewSeverity.WARNING),
    (r"Check out", "checkout", ReviewSeverity.WARNING),
    (r"Continue shopping", "shopify_cart", ReviewSeverity.WARNING),
    (r"\[Judge\.me\]", "judge_me", ReviewSeverity.WARNING),
    (r"Verified\s*\]", "review_widget", ReviewSeverity.WARNING),
    (r"Payment methods", "payment_ui", ReviewSeverity.WARNING),
    (r"##\s+About\b", "footer", ReviewSeverity.INFO),
    (r"##\s+Help\b", "footer", ReviewSeverity.INFO),
    (r"##\s+Get In Touch", "footer", ReviewSeverity.INFO),
    (r"Taxes and shipping calculated at checkout", "shopify_cart", ReviewSeverity.WARNING),
]


def _website_patterns(website: str) -> list[tuple[str, str, ReviewSeverity]]:
    if website == "earkart.in":
        return EARKART_IN_PATTERNS
    if website == "earkart.com":
        return EARKART_COM_PATTERNS
    return []


def _location_hint(text: str, start: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()[:200]


def scan_boilerplate(bundle: DocumentBundle) -> list[BoilerplateMatch]:
    canonical = bundle.canonical
    text = bundle.cleaned.content if bundle.cleaned else (canonical.plain_text or "")
    if not text:
        return []

    matches: list[BoilerplateMatch] = []
    for pattern, category, severity in _website_patterns(canonical.website):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(
                BoilerplateMatch(
                    document_id=canonical.document_id,
                    website=canonical.website,
                    url=canonical.canonical_url,
                    title=canonical.title,
                    pattern=pattern,
                    category=category,
                    location=f"offset {match.start()}",
                    severity=severity,
                    context=_location_hint(text, match.start()),
                )
            )
    return matches


def detect_url_duplicates(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    """Exact canonical URL duplicates (multiple document_ids for same URL)."""
    by_url: dict[str, list[DocumentBundle]] = {}
    for bundle in bundles:
        key = f"{bundle.canonical.website}:{bundle.canonical.canonical_url}"
        by_url.setdefault(key, []).append(bundle)

    groups = []
    for key, group in by_url.items():
        if len(group) > 1:
            groups.append(
                {
                    "type": "exact_url_duplicate",
                    "key": key,
                    "documents": [
                        {
                            "document_id": b.canonical.document_id,
                            "title": b.canonical.title,
                            "content_hash": b.canonical.content_hash,
                        }
                        for b in group
                    ],
                }
            )
    return {"exact_url_duplicates": groups}


def detect_content_hash_duplicates(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    by_hash: dict[str, list[DocumentBundle]] = {}
    for bundle in bundles:
        by_hash.setdefault(bundle.canonical.content_hash, []).append(bundle)

    groups = []
    for content_hash, group in by_hash.items():
        if len(group) > 1:
            groups.append(
                {
                    "type": "exact_content_hash_duplicate",
                    "content_hash": content_hash,
                    "documents": [
                        {
                            "document_id": b.canonical.document_id,
                            "website": b.canonical.website,
                            "url": b.canonical.canonical_url,
                            "title": b.canonical.title,
                        }
                        for b in group
                    ],
                }
            )
    return {"exact_content_hash_duplicates": groups}


def detect_pdf_url_duplicates(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    by_hash: dict[str, list[DocumentBundle]] = {}
    for bundle in bundles:
        if bundle.canonical.source_type.value in {"pdf", "scanned_pdf"}:
            raw_hash = bundle.canonical.metadata.get("raw_content_hash") or bundle.canonical.content_hash
            by_hash.setdefault(raw_hash, []).append(bundle)

    groups = []
    for raw_hash, group in by_hash.items():
        urls = {b.canonical.canonical_url for b in group}
        if len(urls) > 1:
            groups.append(
                {
                    "type": "same_pdf_multiple_urls",
                    "raw_content_hash": raw_hash,
                    "documents": [
                        {
                            "document_id": b.canonical.document_id,
                            "url": b.canonical.canonical_url,
                            "title": b.canonical.title,
                        }
                        for b in group
                    ],
                }
            )
    return {"same_pdf_multiple_urls": groups}


def detect_index_html_duplicates(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    normalized_groups: dict[str, list[DocumentBundle]] = {}
    for bundle in bundles:
        url = bundle.canonical.source_url or bundle.canonical.canonical_url
        normalized_groups.setdefault(normalize_url(url), []).append(bundle)

    groups = []
    for normalized, group in normalized_groups.items():
        raw_urls = {b.canonical.source_url for b in group}
        if len(group) > 1 and len(raw_urls) > 1:
            has_index = any(
                any(seg.lower() in INDEX_ALIASES for seg in urlparse(u).path.split("/") if seg)
                for u in raw_urls
                if u
            )
            if has_index or len({b.canonical.canonical_url for b in group}) == 1:
                groups.append(
                    {
                        "type": "index_html_alias",
                        "normalized_url": normalized,
                        "documents": [
                            {
                                "document_id": b.canonical.document_id,
                                "source_url": b.canonical.source_url,
                                "canonical_url": b.canonical.canonical_url,
                            }
                            for b in group
                        ],
                    }
                )
    return {"index_html_aliases": groups}


def detect_query_param_duplicates(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    by_canonical: dict[str, list[DocumentBundle]] = {}
    for bundle in bundles:
        by_canonical.setdefault(bundle.canonical.canonical_url, []).append(bundle)

    groups = []
    for canonical_url, group in by_canonical.items():
        source_urls = {b.canonical.source_url for b in group if b.canonical.source_url}
        if len(source_urls) > 1:
            groups.append(
                {
                    "type": "query_param_duplicate",
                    "canonical_url": canonical_url,
                    "source_urls": sorted(source_urls),
                    "documents": [
                        {"document_id": b.canonical.document_id, "source_url": b.canonical.source_url}
                        for b in group
                    ],
                }
            )
    return {"query_param_duplicates": groups}


def detect_cross_site_duplicates(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    by_raw_hash: dict[str, list[DocumentBundle]] = {}
    for bundle in bundles:
        raw_hash = bundle.canonical.metadata.get("raw_content_hash")
        if raw_hash:
            by_raw_hash.setdefault(raw_hash, []).append(bundle)

    groups = []
    for raw_hash, group in by_raw_hash.items():
        websites = {b.canonical.website for b in group}
        if len(websites) > 1:
            groups.append(
                {
                    "type": "cross_site_identical_content",
                    "raw_content_hash": raw_hash,
                    "websites": sorted(websites),
                    "documents": [
                        {
                            "document_id": b.canonical.document_id,
                            "website": b.canonical.website,
                            "url": b.canonical.canonical_url,
                            "title": b.canonical.title,
                        }
                        for b in group
                    ],
                }
            )
    return {"cross_site_identical_content": groups}


def run_duplicate_analysis(bundles: list[DocumentBundle]) -> dict[str, list[dict]]:
    report: dict[str, list[dict]] = {}
    for section in (
        detect_url_duplicates,
        detect_content_hash_duplicates,
        detect_pdf_url_duplicates,
        detect_index_html_duplicates,
        detect_query_param_duplicates,
        detect_cross_site_duplicates,
    ):
        report.update(section(bundles))
    return report
