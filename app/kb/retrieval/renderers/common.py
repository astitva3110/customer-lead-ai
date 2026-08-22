"""Shared renderer utilities."""

import re

FAQ_QUESTION = re.compile(r"^Q(\d+)\.\s+(.+)$", re.IGNORECASE)

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
PIPELINE_HASH = re.compile(r"sha256:[a-f0-9]{64}", re.I)
SHOPIFY_FORM_UI = re.compile(
    r"^(Name\*?|Email\*?|Phone number\*?|Your City/ZIP/PIN Code|Comment|Send|Send Message|Message Us|"
    r"Drop Us Message for Any Query|Contact form|Let's get in touch|Skip to content|Add to cart|"
    r"Filter and sort|Taxes and shipping calculated at checkout)$",
    re.I,
)
NAV_LINK_URL_MARKERS = (
    "/products/",
    "/collections/",
    "/cart",
    "/account",
    "/cdn/shop/",
    "cdn.shopify.com",
    "/blogs/",
    "flogo.webp",
    "index.html",
)


def clean_title(title: str) -> str:
    return " ".join(title.split())


def join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(block for block in blocks if block.strip())


def _is_navigation_link(label: str, url: str) -> bool:
    lowered_url = url.lower()
    cleaned_label = label.strip().lower()
    if cleaned_label in {"view", "order now", "save your money now", "save upto ₹ 25,000"}:
        return True
    if any(marker in lowered_url for marker in NAV_LINK_URL_MARKERS):
        return True
    if cleaned_label.startswith("![]"):
        return True
    return False


def sanitize_block_text(text: str) -> str:
    """Remove navigation markdown and UI-only lines from one content block."""
    if not text.strip():
        return ""

    text = MARKDOWN_IMAGE.sub("", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        url = match.group(2)
        if _is_navigation_link(label, url):
            cleaned = re.sub(r"\s+View\s*$", "", label, flags=re.I).strip()
            if cleaned.lower() in {"view", "order now"} or not cleaned:
                return ""
            if any(marker in url.lower() for marker in ("/products/", "/pages/earkart-benefits")):
                return ""
            return cleaned
        return label

    text = MARKDOWN_LINK.sub(replace_link, text)
    text = PIPELINE_HASH.sub("", text)

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if SHOPIFY_FORM_UI.match(stripped):
            continue
        if stripped.startswith("- [") and "](http" in stripped:
            continue
        kept_lines.append(stripped)

    return "\n".join(kept_lines).strip()


def sanitize_retrieval_text(text: str) -> str:
    """Final retrieval_text cleanup: navigation noise and pipeline metadata."""
    blocks = [sanitize_block_text(block) for block in text.split("\n\n")]
    cleaned = join_blocks(blocks)
    cleaned = PIPELINE_HASH.sub("", cleaned)
    return cleaned.strip()
