"""Conservative document type classification for retrieval preparation."""

from app.kb.enums import DocumentType, SourceType
from app.kb.models.structured_content import DocumentContent


def _has_faq_structure(content: DocumentContent) -> bool:
    faq_markers = ("faq", "frequently asked")
    title = (content.title or "").lower()
    if any(marker in title for marker in faq_markers):
        return True

    def walk(nodes) -> bool:
        for node in nodes:
            node_type = getattr(node, "type", None)
            if node_type == "heading":
                text = node.text.lower()
                if any(marker in text for marker in faq_markers):
                    return True
            elif node_type == "section":
                heading = (node.heading or "").lower()
                if any(marker in heading for marker in faq_markers):
                    return True
                if walk(node.children):
                    return True
            elif node_type == "paragraph" and node.text.strip().upper().startswith("FAQ"):
                return True
        return False

    if content.pages:
        for page in content.pages:
            if walk(page.blocks):
                return True
        return False
    return walk(content.children)


def classify_document(
    *,
    canonical_url: str,
    title: str,
    source_type: SourceType,
    structured_content: DocumentContent,
) -> DocumentType:
    """Classify document type from URL/title/structure without modifying content."""
    url = canonical_url.lower()
    title_lower = title.lower()

    if "/products/" in url:
        return DocumentType.PRODUCT

    policy_slugs = (
        "terms",
        "privacy",
        "return",
        "warranty",
        "policy",
        "shipping",
        "refund",
    )
    if "/pages/" in url and any(slug in url for slug in policy_slugs):
        return DocumentType.POLICY
    if any(token in title_lower for token in ("terms & conditions", "privacy policy", "return", "warranty")):
        return DocumentType.POLICY

    if _has_faq_structure(structured_content) or "faq" in url:
        return DocumentType.FAQ

    if "/blog/" in url or "blog" in title_lower:
        return DocumentType.BLOG

    if "/investor/" in url:
        notice_tokens = ("notice", "agenda", "outcome", "pre-intimation", "/gm/", "/bt/", "/notices/")
        if any(token in url for token in notice_tokens):
            return DocumentType.NOTICE
        return DocumentType.INVESTOR

    if "/radius/" in url:
        return DocumentType.PRODUCT

    if source_type in (SourceType.PDF, SourceType.SCANNED_PDF):
        product_tokens = ("radius", "eqfy", "tiny", "fame", "bluup", "specification")
        if any(token in url for token in product_tokens):
            return DocumentType.PRODUCT
        if "/investor/" in url:
            return DocumentType.INVESTOR
        return DocumentType.OTHER

    if source_type == SourceType.HTML:
        return DocumentType.WEBPAGE

    return DocumentType.OTHER
