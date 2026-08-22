import re

from app.kb.cleaning.base import CleaningResult
from app.kb.cleaning.generic import GenericCleaner, normalize_whitespace
from app.kb.models.raw import RawArtifact
from app.kb.models.structured_content import PageContent, ParagraphNode

_ISOLATED_CHAR_RE = re.compile(r"^[.|\\/_\-=<>]{1,3}$")
_BROKEN_WORD_SPACE_RE = re.compile(r"([a-z])([A-Z])")
_ACT_REF_RE = re.compile(r"Act\.(\d)")
_SECTION_REF_RE = re.compile(r"Section\s+(\d)")


def _normalize_ocr_line(line: str) -> str:
    stripped = line.strip()
    if _ISOLATED_CHAR_RE.match(stripped):
        return ""
    stripped = _ACT_REF_RE.sub(r"Act. \1", stripped)
    stripped = _SECTION_REF_RE.sub(r"Section \1", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped


def _clean_ocr_text(text: str) -> str:
    lines = [_normalize_ocr_line(line) for line in text.split("\n")]
    lines = [line for line in lines if line]
    return normalize_whitespace("\n".join(lines))


class OcrCleaner:
    profile = "generic.ocr"
    profile_version = "1.0"

    def __init__(self, profile: str | None = None) -> None:
        if profile:
            self.profile = f"{profile}.ocr"

    def clean(self, artifact: RawArtifact) -> CleaningResult:
        from app.kb.cleaning.pdf import PdfCleaner, _infer_blocks, _strip_pdf_wrapper

        generic = GenericCleaner()
        removed = ["ocr_noise_lines"]

        if artifact.pages:
            cleaned_pages: list[PageContent] = []
            page_texts: list[str] = []
            for page in artifact.pages:
                raw = "\n".join(
                    block.text for block in page.blocks if hasattr(block, "text") and block.text
                )
                raw = _strip_pdf_wrapper(raw)
                cleaned = _clean_ocr_text(raw)
                page_texts.append(cleaned)
                blocks = _infer_blocks(cleaned) or [ParagraphNode(text=cleaned, confidence=page.ocr_confidence)]
                for block in blocks:
                    if hasattr(block, "confidence"):
                        block.confidence = page.ocr_confidence
                cleaned_pages.append(
                    PageContent(
                        page_number=page.page_number,
                        ocr_confidence=page.ocr_confidence,
                        blocks=blocks,
                    )
                )
            content = "\n\n".join(page_texts)
        else:
            content = _clean_ocr_text(_strip_pdf_wrapper(artifact.content))
            cleaned_pages = [
                PageContent(page_number=1, blocks=_infer_blocks(content) or [ParagraphNode(text=content)])
            ]

        content, generic_removed = generic.clean_text(content)
        removed.extend(generic_removed)

        return CleaningResult(
            content=content,
            pages=cleaned_pages,
            ocr=artifact.ocr,
            profile=self.profile,
            profile_version=self.profile_version,
            removed_elements=sorted(set(removed)),
        )
