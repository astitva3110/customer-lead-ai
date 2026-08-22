import re

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_IMAGE_ONLY_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")
_LINK_ONLY_IMAGE_RE = re.compile(r"^\[\!\[[^\]]*\]\([^)]+\)\]\([^)]+\)\s*$")
_EMPTY_LINK_RE = re.compile(r"^\[\s*\]\([^)]+\)\s*$")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_html_artifacts(text: str) -> str:
    text = _HTML_COMMENT_RE.sub("", text)
    text = _SCRIPT_STYLE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return text


def remove_empty_markdown_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _IMAGE_ONLY_LINE_RE.match(stripped):
            continue
        if _LINK_ONLY_IMAGE_RE.match(stripped):
            continue
        if _EMPTY_LINK_RE.match(stripped):
            continue
        kept.append(line.rstrip())
    return normalize_whitespace("\n".join(kept))


class GenericCleaner:
    """Reusable markdown normalization applied before site-specific rules."""

    profile = "generic"
    profile_version = "1.0"

    def clean_text(self, text: str) -> tuple[str, list[str]]:
        removed: list[str] = []
        original = text
        text = strip_html_artifacts(text)
        if text != original:
            removed.append("html_artifacts")

        text = remove_empty_markdown_lines(text)
        text = normalize_whitespace(text)
        return text, removed
