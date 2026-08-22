import re
from collections import Counter

from app.kb.cleaning.base import CleaningResult
from app.kb.cleaning.generic import GenericCleaner, normalize_whitespace
from app.kb.models.raw import RawArtifact
from app.kb.models.structured_content import PageContent, ParagraphNode

_PDF_WRAPPER_RE = re.compile(r"^#\s+.+\n\nSource:\s+https?://[^\n]+\n\n", re.DOTALL)
_HEADING_LINE_RE = re.compile(r"^[A-Z][A-Z0-9 \-/&(),.'\"]{3,}$")

# Repeated company letterhead seen across earkart.in PDFs.
_KNOWN_FOOTER_PATTERNS = (
    re.compile(r"^A-133, Ground Floor.*Noida.*201301\s*$", re.IGNORECASE),
    re.compile(r"^Contact Number\s*:.*info@earkart\.in\s*$", re.IGNORECASE),
    re.compile(r"^website\s*:\s*www\.earkart\.in\s*$", re.IGNORECASE),
    re.compile(r"^earKART\s+Private Limited\s*$", re.IGNORECASE),
    re.compile(r"^Registered Address:.*Delhi-110096\s*$", re.IGNORECASE),
    re.compile(r"^Corporate Address:.*Uttar Pradesh.*201301\s*$", re.IGNORECASE),
    re.compile(r"^www\.earkart\.in\s*$", re.IGNORECASE),
    re.compile(r"^info@earkart\.in\s*$", re.IGNORECASE),
    re.compile(r"^0120[- ]?4102857\s*$", re.IGNORECASE),
    re.compile(r"^REDEFINING\s*$", re.IGNORECASE),
    re.compile(r"^HEARING\s*$", re.IGNORECASE),
    re.compile(r"^CARE\s*$", re.IGNORECASE),
)


def _strip_pdf_wrapper(text: str) -> str:
    return _PDF_WRAPPER_RE.sub("", text.strip(), count=1)


def _detect_repeated_lines(pages: list[str], min_pages: int = 2) -> set[str]:
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        for line in page.split("\n"):
            normalized = line.strip()
            if normalized:
                counts[normalized] += 1
    threshold = max(2, len(pages) // 2)
    return {line for line, count in counts.items() if count >= threshold}


def _is_known_footer(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in _KNOWN_FOOTER_PATTERNS)


def _clean_page_text(text: str, repeated: set[str]) -> tuple[str, list[str], list[str]]:
    removed_headers: list[str] = []
    removed_footers: list[str] = []
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if stripped in repeated or _is_known_footer(stripped):
            if _is_known_footer(stripped) or stripped in repeated:
                removed_footers.append(stripped)
            continue
        kept.append(stripped)
    return normalize_whitespace("\n".join(kept)), removed_headers, removed_footers


def _infer_blocks(text: str) -> list:
    blocks: list = []
    paragraph: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if paragraph:
                blocks.append(ParagraphNode(text="\n".join(paragraph)))
                paragraph = []
            continue
        if _HEADING_LINE_RE.match(stripped) and len(stripped.split()) <= 12:
            if paragraph:
                blocks.append(ParagraphNode(text="\n".join(paragraph)))
                paragraph = []
            from app.kb.models.structured_content import HeadingNode

            blocks.append(HeadingNode(text=stripped, level=2))
            continue
        paragraph.append(stripped)
    if paragraph:
        blocks.append(ParagraphNode(text="\n".join(paragraph)))
    return blocks


class PdfCleaner:
    profile = "generic.pdf"
    profile_version = "1.0"

    def __init__(self, profile: str | None = None) -> None:
        if profile:
            self.profile = f"{profile}.pdf"

    def clean(self, artifact: RawArtifact) -> CleaningResult:
        generic = GenericCleaner()
        removed: list[str] = []
        removed_headers: list[str] = []
        removed_footers: list[str] = []

        if artifact.pages:
            raw_pages = []
            for page in artifact.pages:
                page_text = "\n".join(
                    block.text for block in page.blocks if hasattr(block, "text") and block.text
                )
                raw_pages.append(_strip_pdf_wrapper(page_text))
            repeated = _detect_repeated_lines(raw_pages)
            cleaned_pages: list[PageContent] = []
            for page, raw_text in zip(artifact.pages, raw_pages, strict=True):
                cleaned_text, page_headers, page_footers = _clean_page_text(raw_text, repeated)
                removed_headers.extend(page_headers)
                removed_footers.extend(page_footers)
                cleaned_pages.append(
                    PageContent(
                        page_number=page.page_number,
                        blocks=_infer_blocks(cleaned_text) or [ParagraphNode(text=cleaned_text)],
                    )
                )
            content = "\n\n".join(
                f"--- Page {page.page_number} ---\n"
                + "\n".join(block.text for block in page.blocks if hasattr(block, "text"))
                for page in cleaned_pages
            )
            if repeated:
                removed.append("repeated_header_footer")
            return CleaningResult(
                content=generic.clean_text(content)[0],
                pages=cleaned_pages,
                profile=self.profile,
                profile_version=self.profile_version,
                removed_elements=sorted(set(removed)),
                removed_headers=removed_headers[:20],
                removed_footers=removed_footers[:20],
            )

        text = _strip_pdf_wrapper(artifact.content)
        text, generic_removed = generic.clean_text(text)
        removed.extend(generic_removed)
        if _PDF_WRAPPER_RE.match(artifact.content.strip()):
            removed.append("pdf_wrapper")

        return CleaningResult(
            content=text,
            pages=[PageContent(page_number=1, blocks=_infer_blocks(text) or [ParagraphNode(text=text)])],
            profile=self.profile,
            profile_version=self.profile_version,
            removed_elements=sorted(set(removed)),
        )
