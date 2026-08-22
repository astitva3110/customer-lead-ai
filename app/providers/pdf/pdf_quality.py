"""PDF extraction quality assessment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

MEANINGFUL_CHAR_RE = re.compile(r"[\w]", re.UNICODE)

# Calibrated from existing RAW PDF dataset (79 native PDFs):
# - median meaningful chars: 2219; sparse failures cluster at 14-60 chars
# - image-heavy CorelDRAW spec sheets expose a tiny text layer plus large images
MIN_MEANINGFUL_CHARS_PER_PAGE = 150
MIN_WORDS_PER_PAGE = 20
IMAGE_HEAVY_AREA_RATIO = 0.10
IMAGE_DOMINANT_AREA_RATIO = 0.25
VECTOR_HEAVY_DRAWING_COUNT = 50
VECTOR_DENSE_VISUAL_RATIO = 0.25


@dataclass
class PageQualityMetrics:
    page_number: int
    char_count: int
    word_count: int
    line_count: int
    block_count: int
    image_area_ratio: float
    text_density: float
    drawing_count: int = 0
    visual_density: float = 0.0


@dataclass
class ExtractionQualityReport:
    page_count: int
    total_chars: int
    total_words: int
    total_lines: int
    total_blocks: int
    avg_image_area_ratio: float
    is_suspiciously_sparse: bool
    is_image_heavy: bool
    quality_score: float
    pages: list[PageQualityMetrics] = field(default_factory=list)


def meaningful_char_count(text: str) -> int:
    return len(MEANINGFUL_CHAR_RE.findall(text))


def _page_image_area_ratio(page: pymupdf.Page) -> float:
    page_area = max(page.rect.width * page.rect.height, 1.0)
    image_area = 0.0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        x0, y0, x1, y1 = block["bbox"]
        image_area += max(0.0, (x1 - x0) * (y1 - y0))
    return min(image_area / page_area, 1.0)


def _page_visual_density(page: pymupdf.Page, dpi: int = 36) -> float:
    """Estimate non-background pixel ratio as a proxy for visual content density."""
    pixmap = page.get_pixmap(dpi=dpi)
    samples = pixmap.samples
    channels = min(3, pixmap.n)
    nonwhite = 0
    for index in range(0, len(samples), pixmap.n):
        if any(samples[index + channel] < 250 for channel in range(channels)):
            nonwhite += 1
    return nonwhite / max(pixmap.width * pixmap.height, 1)


def compute_page_metrics(page: pymupdf.Page, page_number: int, text: str) -> PageQualityMetrics:
    lines = [line for line in text.splitlines() if line.strip()]
    words = re.findall(r"\S+", text)
    blocks = page.get_text("blocks")
    text_blocks = [block for block in blocks if isinstance(block, tuple) and len(block) >= 5]
    page_area = max(page.rect.width * page.rect.height, 1.0)
    char_count = meaningful_char_count(text)
    drawing_count = len(page.get_drawings())
    visual_density = _page_visual_density(page)
    return PageQualityMetrics(
        page_number=page_number,
        char_count=char_count,
        word_count=len(words),
        line_count=len(lines),
        block_count=len(text_blocks),
        image_area_ratio=_page_image_area_ratio(page),
        text_density=char_count / page_area * 1000.0,
        drawing_count=drawing_count,
        visual_density=visual_density,
    )


def _compute_quality_score(report: ExtractionQualityReport) -> float:
    if report.page_count == 0:
        return 0.0

    chars_per_page = report.total_chars / report.page_count
    words_per_page = report.total_words / report.page_count

    char_score = min(chars_per_page / 500.0, 1.0) * 50.0
    word_score = min(words_per_page / 80.0, 1.0) * 30.0
    block_score = min(report.total_blocks / max(report.page_count * 3, 1), 1.0) * 10.0
    image_penalty = report.avg_image_area_ratio * 10.0 if chars_per_page < MIN_MEANINGFUL_CHARS_PER_PAGE else 0.0

    return max(0.0, min(100.0, char_score + word_score + block_score - image_penalty))


def assess_extraction_quality(doc: pymupdf.Document, pages_text: list[str]) -> ExtractionQualityReport:
    page_metrics: list[PageQualityMetrics] = []
    for index, text in enumerate(pages_text, start=1):
        page = doc[index - 1]
        page_metrics.append(compute_page_metrics(page, index, text))

    total_chars = sum(page.char_count for page in page_metrics)
    total_words = sum(page.word_count for page in page_metrics)
    total_lines = sum(page.line_count for page in page_metrics)
    total_blocks = sum(page.block_count for page in page_metrics)
    avg_image_area = (
        sum(page.image_area_ratio for page in page_metrics) / len(page_metrics) if page_metrics else 0.0
    )

    report = ExtractionQualityReport(
        page_count=len(page_metrics),
        total_chars=total_chars,
        total_words=total_words,
        total_lines=total_lines,
        total_blocks=total_blocks,
        avg_image_area_ratio=avg_image_area,
        is_suspiciously_sparse=False,
        is_image_heavy=avg_image_area >= IMAGE_HEAVY_AREA_RATIO,
        quality_score=0.0,
        pages=page_metrics,
    )
    report.is_suspiciously_sparse = is_poor_quality(report)
    report.quality_score = _compute_quality_score(report)
    return report


def is_poor_quality(report: ExtractionQualityReport) -> bool:
    """Return True when native extraction likely missed substantial page content."""
    if report.page_count == 0 or report.total_chars == 0:
        return True

    chars_per_page = report.total_chars / report.page_count
    words_per_page = report.total_words / report.page_count

    # Calibrated: every native PDF under 150 meaningful chars in the dataset is a
    # CorelDRAW vector spec sheet with a misleading partial text layer.
    if chars_per_page < MIN_MEANINGFUL_CHARS_PER_PAGE:
        return True

    if report.is_image_heavy and chars_per_page < MIN_MEANINGFUL_CHARS_PER_PAGE * 2:
        return True

    if report.avg_image_area_ratio >= IMAGE_DOMINANT_AREA_RATIO and words_per_page < MIN_WORDS_PER_PAGE:
        return True

    for page in report.pages:
        if page.image_area_ratio >= IMAGE_HEAVY_AREA_RATIO and page.char_count < MIN_MEANINGFUL_CHARS_PER_PAGE:
            return True
        if page.image_area_ratio >= IMAGE_DOMINANT_AREA_RATIO and page.word_count < MIN_WORDS_PER_PAGE:
            return True
        if (
            page.drawing_count >= VECTOR_HEAVY_DRAWING_COUNT
            and page.char_count < MIN_MEANINGFUL_CHARS_PER_PAGE * 2
            and page.visual_density >= VECTOR_DENSE_VISUAL_RATIO
        ):
            return True

    return False


def is_better_quality(candidate: ExtractionQualityReport, current: ExtractionQualityReport) -> bool:
    if candidate.total_chars != current.total_chars:
        return candidate.total_chars > current.total_chars
    if candidate.total_words != current.total_words:
        return candidate.total_words > current.total_words
    return candidate.quality_score > current.quality_score
