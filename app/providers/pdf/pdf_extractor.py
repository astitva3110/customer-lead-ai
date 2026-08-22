import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pymupdf

from app.kb.enums import ExtractionMethod
from app.kb.models.structured_content import ContentNode, OcrMetadata, ParagraphNode
from app.providers.pdf.pdf_layout import extract_page_layout
from app.providers.pdf.pdf_quality import (
    ExtractionQualityReport,
    assess_extraction_quality,
    is_better_quality,
    is_poor_quality,
)

PDF_URL_RE = re.compile(r"https?://[^\s\)\"'<>]+\.pdf(?:[^\s\)\"'<>]*)?", re.IGNORECASE)

TESSERACT_DIRS = (
    Path(r"C:\Program Files\Tesseract-OCR"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR"),
)


@dataclass
class PdfPageResult:
    page_number: int
    text: str
    blocks: list[ContentNode] = field(default_factory=list)
    ocr_confidence: float | None = None


@dataclass
class PdfExtractResult:
    title: str
    pages: list[PdfPageResult] = field(default_factory=list)
    extraction_method: ExtractionMethod = ExtractionMethod.PDF_TEXT
    ocr: OcrMetadata | None = None
    full_text: str = ""
    quality: ExtractionQualityReport | None = None
    fallback_used: bool = False


def _configure_tesseract() -> bool:
    for directory in TESSERACT_DIRS:
        executable = directory / "tesseract.exe"
        if executable.exists():
            tessdata = directory / "tessdata"
            os.environ["TESSDATA_PREFIX"] = str(tessdata)
            path_prefix = str(directory)
            current_path = os.environ.get("PATH", "")
            if path_prefix not in current_path:
                os.environ["PATH"] = f"{path_prefix}{os.pathsep}{current_path}"
            return True
    return False


def discover_pdf_urls(texts: list[str], allowed_domains: list[str]) -> list[str]:
    domains = {domain.replace("www.", "").lower() for domain in allowed_domains}
    found: set[str] = set()

    for text in texts:
        for match in PDF_URL_RE.findall(text):
            url = match.rstrip(".,);]")
            host = urlparse(url).netloc.replace("www.", "").lower()
            if host in domains:
                found.add(url)

    return sorted(found)


def _extract_native_simple(doc: pymupdf.Document) -> list[PdfPageResult]:
    pages: list[PdfPageResult] = []
    for index, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append(
                PdfPageResult(
                    page_number=index,
                    text=text,
                    blocks=[ParagraphNode(text=text)],
                )
            )
    return pages


def _extract_native_layout(doc: pymupdf.Document) -> list[PdfPageResult]:
    pages: list[PdfPageResult] = []
    for index, page in enumerate(doc, start=1):
        text, blocks = extract_page_layout(page)
        if text:
            pages.append(PdfPageResult(page_number=index, text=text, blocks=blocks))
    return pages


def _extract_pages_with_ocr(doc: pymupdf.Document, dpi: int = 200) -> list[PdfPageResult]:
    if not _configure_tesseract():
        return []

    pages: list[PdfPageResult] = []
    for index, page in enumerate(doc, start=1):
        try:
            textpage = page.get_textpage_ocr(dpi=dpi, full=True, language="eng")
            text, blocks = extract_page_layout(page, textpage=textpage)
            if not text:
                text = page.get_text(textpage=textpage).strip()
                blocks = [ParagraphNode(text=text)] if text else []
        except Exception:
            text = ""
            blocks = []
        if text:
            pages.append(PdfPageResult(page_number=index, text=text, blocks=blocks))
    return pages


def _pages_to_result(
    *,
    title: str,
    pages: list[PdfPageResult],
    doc: pymupdf.Document,
    extraction_method: ExtractionMethod,
    ocr: OcrMetadata | None = None,
    fallback_used: bool = False,
) -> PdfExtractResult:
    pages_text = [page.text for page in pages]
    quality = assess_extraction_quality(doc, pages_text) if pages_text else None
    full_text = "\n\n".join(page.text for page in pages)
    return PdfExtractResult(
        title=title,
        pages=pages,
        extraction_method=extraction_method,
        ocr=ocr,
        full_text=full_text,
        quality=quality,
        fallback_used=fallback_used,
    )


def _evaluate_candidate(
    *,
    title: str,
    pages: list[PdfPageResult],
    doc: pymupdf.Document,
    extraction_method: ExtractionMethod,
    ocr: OcrMetadata | None = None,
    fallback_used: bool = False,
) -> PdfExtractResult:
    if not pages:
        return PdfExtractResult(title=title, pages=[], extraction_method=extraction_method, ocr=ocr)
    return _pages_to_result(
        title=title,
        pages=pages,
        doc=doc,
        extraction_method=extraction_method,
        ocr=ocr,
        fallback_used=fallback_used,
    )


def extract_pdf_bytes(
    pdf_bytes: bytes,
    *,
    title: str | None = None,
    ocr_fallback: bool = True,
    ocr_dpi: int = 200,
) -> PdfExtractResult:
    """Extract PDF content from raw bytes with quality-checked fallback chain."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        resolved_title = title or (doc.metadata or {}).get("title") or "Untitled"

        simple_pages = _extract_native_simple(doc)
        simple_result = _evaluate_candidate(
            title=resolved_title,
            pages=simple_pages,
            doc=doc,
            extraction_method=ExtractionMethod.PDF_TEXT,
        )

        if simple_result.quality and not is_poor_quality(simple_result.quality):
            return simple_result

        layout_pages = _extract_native_layout(doc)
        layout_result = _evaluate_candidate(
            title=resolved_title,
            pages=layout_pages,
            doc=doc,
            extraction_method=ExtractionMethod.PDF_TEXT,
            fallback_used=bool(layout_pages),
        )

        best = simple_result
        if layout_result.quality and (
            best.quality is None or is_better_quality(layout_result.quality, best.quality)
        ):
            best = layout_result

        if best.quality and not is_poor_quality(best.quality):
            return best

        if ocr_fallback:
            ocr_pages = _extract_pages_with_ocr(doc, dpi=ocr_dpi)
            ocr_result = _evaluate_candidate(
                title=resolved_title,
                pages=ocr_pages,
                doc=doc,
                extraction_method=ExtractionMethod.OCR,
                ocr=OcrMetadata(engine="tesseract", dpi=ocr_dpi, language="eng"),
                fallback_used=True,
            )
            if ocr_result.quality and (
                best.quality is None or is_better_quality(ocr_result.quality, best.quality)
            ):
                return ocr_result

        return best if best.pages else PdfExtractResult(title=resolved_title, pages=[], extraction_method=ExtractionMethod.PDF_TEXT)
    finally:
        doc.close()


def extract_pdf(
    url: str,
    client: httpx.Client,
    *,
    ocr_fallback: bool = True,
    ocr_dpi: int = 200,
) -> PdfExtractResult:
    """Extract PDF with per-page structure, quality assessment, and fallback."""
    response = client.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    title = urlparse(url).path.split("/")[-1] or url
    return extract_pdf_bytes(
        response.content,
        title=title,
        ocr_fallback=ocr_fallback,
        ocr_dpi=ocr_dpi,
    )


def extract_pdf_text(
    url: str,
    client: httpx.Client,
    *,
    ocr_fallback: bool = True,
    ocr_dpi: int = 200,
) -> tuple[str, str]:
    """Backward-compatible wrapper returning (title, body)."""
    result = extract_pdf(url, client, ocr_fallback=ocr_fallback, ocr_dpi=ocr_dpi)
    return result.title, result.full_text
