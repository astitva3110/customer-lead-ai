"""Website inclusion policy and document exclusion rules."""

from urllib.parse import urlparse

from app.kb.enums import ExclusionReason, WebsitePolicy
from app.kb.models.raw import RawArtifact

TARGET_WEBSITES = frozenset({"earkart.in", "earkart.com"})

NON_TARGET_WEBSITES = frozenset({"crm.earkart.in", "firecrawl.dev"})

EXCLUDED_URL_PATTERNS: tuple[tuple[str, ExclusionReason], ...] = (
    ("/cart", ExclusionReason.CART_CHECKOUT),
    ("/checkout", ExclusionReason.CART_CHECKOUT),
    ("/apps/", ExclusionReason.THIRD_PARTY_APP),
)

# Explicit production KB exclusions — RAW artifacts preserved, not downstream eligible.
PRODUCTION_EXCLUDED_CANONICAL_URLS: frozenset[str] = frozenset(
    {
        "https://earkart.in/investor/notices/FE-Delhi-April-02--2026-Earkart.pdf",
        "https://earkart.in/investor/bp/WhistleBlower-Policy.pdf",
        "https://earkart.in/investor/gm/FE-Delhi-June-25-2026.pdf",
        "https://earkart.in/investor/gm/JS-Delhi-25-June-2026.pdf",
        "https://earkart.in/investor/bp/Code-of-Conduct-Policy.pdf",
        "https://earkart.in/ZB.pdf",
    }
)


def website_policy(website: str) -> WebsitePolicy:
    normalized = website.lower().removeprefix("www.")
    if normalized in TARGET_WEBSITES:
        return WebsitePolicy.TARGET
    return WebsitePolicy.NON_TARGET


def is_target_website(website: str) -> bool:
    return website_policy(website) == WebsitePolicy.TARGET


def detect_exclusion(raw: RawArtifact) -> ExclusionReason | None:
    website = raw.website.lower().removeprefix("www.")
    if website in NON_TARGET_WEBSITES:
        return ExclusionReason.NON_TARGET_WEBSITE
    if website not in TARGET_WEBSITES:
        return ExclusionReason.NON_TARGET_WEBSITE

    if raw.canonical_url in PRODUCTION_EXCLUDED_CANONICAL_URLS:
        return ExclusionReason.NOT_REQUIRED_FOR_PRODUCTION_KB

    url = raw.canonical_url.lower()
    title = raw.title.lower()

    for pattern, reason in EXCLUDED_URL_PATTERNS:
        if pattern in url:
            return reason

    if "404" in title or "not found" in title:
        return ExclusionReason.NOT_FOUND_404
    if "page not found" in (raw.content or "").lower()[:500]:
        return ExclusionReason.NOT_FOUND_404

    return None


def build_exclusion_metadata(website: str, reason: ExclusionReason) -> dict:
    return {
        "excluded": True,
        "exclusion_reason": reason.value,
        "website_policy": website_policy(website).value,
    }
