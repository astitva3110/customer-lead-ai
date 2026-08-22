"""Selective per-page OCR helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf

from app.providers.pdf.pdf_layout import extract_page_layout
from app.providers.pdf.pdf_quality import MIN_MEANINGFUL_CHARS_PER_PAGE, MIN_WORDS_PER_PAGE, compute_page_metrics
from app.kb.models.structured_content import ParagraphNode


TESSERACT_DIRS = (
    Path(r"C:\Program Files\Tesseract-OCR"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR"),
)


def configure_tesseract() -> bool:
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


def page_needs_ocr(page: pymupdf.Page, native_text: str) -> bool:
    metrics = compute_page_metrics(page, page.number + 1, native_text)
    if metrics.char_count < MIN_MEANINGFUL_CHARS_PER_PAGE:
        return True
    if metrics.image_area_ratio >= 0.10 and metrics.char_count < MIN_MEANINGFUL_CHARS_PER_PAGE * 2:
        return True
    if metrics.image_area_ratio >= 0.25 and metrics.word_count < MIN_WORDS_PER_PAGE:
        return True
    return False


def ocr_page(page: pymupdf.Page, *, dpi: int = 200) -> tuple[str, list]:
    if not configure_tesseract():
        return "", []
    try:
        textpage = page.get_textpage_ocr(dpi=dpi, full=True, language="eng")
        text, blocks = extract_page_layout(page, textpage=textpage)
        if not text:
            text = page.get_text(textpage=textpage).strip()
            blocks = [ParagraphNode(text=text)] if text else []
        return text, blocks
    except Exception:
        return "", []


def extract_native_page(page: pymupdf.Page) -> tuple[str, list]:
    text, blocks = extract_page_layout(page)
    if not text:
        text = page.get_text().strip()
        blocks = [ParagraphNode(text=text)] if text else []
    return text, blocks
