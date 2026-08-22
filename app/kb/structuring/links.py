import re

from app.kb.models.canonical import DocumentLink
from app.kb.url_normalizer import extract_website, normalize_url

_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

NAV_ONLY_LABELS = frozenset(
    {
        "home",
        "about us",
        "contact us",
        "blog",
        "hearing aids",
        "hearing loss",
        "press release",
        "log in",
        "login",
        "create account",
        "continue shopping",
        "check out",
        "skip to content",
        "become our partner",
        "book appointment",
        "cart",
        "search",
    }
)


def extract_links(text: str, website: str) -> list[DocumentLink]:
    links: list[DocumentLink] = []
    seen: set[str] = set()
    for match in _LINK_RE.finditer(text):
        label = match.group(1).strip()
        url = match.group(2).strip()
        if not url or url.startswith("#") or url.startswith("javascript:"):
            continue
        normalized = normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        if label.lower() in NAV_ONLY_LABELS:
            continue
        if not label or label.strip() in {"!", "image", "cart"}:
            continue
        link_host = extract_website(url) if url.startswith("http") else website
        links.append(
            DocumentLink(
                url=normalized,
                text=label or None,
                internal=link_host == website.lower(),
            )
        )
    return links
