"""Deterministic structure recovery for coarse canonical content."""

from __future__ import annotations

import re
from copy import deepcopy

from app.kb.models.structured_content import (
    ContentNode,
    DocumentContent,
    HeadingNode,
    ListNode,
    PageContent,
    ParagraphNode,
    SectionNode,
    TableNode,
)
from app.kb.retrieval.models import StructureRecoveryStats

GIANT_PARAGRAPH_MIN_CHARS = 400

MARKDOWN_BOLD_HEADING = re.compile(r"^\*\*(.+?)\*\*\s*$")
NUMBERED_SECTION_HEADING = re.compile(r"^(\d+)\.\s+(.+)$")
CLAUSE_THREE_LEVEL = re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$")
CLAUSE_TWO_LEVEL = re.compile(r"^(\d+\.\d+)\s+(.+)$")
FAQ_QUESTION = re.compile(r"^Q(\d+)\.\s+(.+)$", re.IGNORECASE)

# PDF/OCR layout signals
ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9\s&,\-./():'\"]{3,80}$")
ROMAN_SECTION = re.compile(r"^(PART|SECTION|CHAPTER|ANNEXURE|APPENDIX|SCHEDULE)\s+[IVXLC\d]+", re.I)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9].{2,})$")
BULLET_LINE = re.compile(r"^[\u2022\-\*•]\s+(.+)$")
PROSPECTUS_URL_MARKER = "/investor/ipo/"


def _is_giant_paragraph(text: str) -> bool:
    stripped = text.strip()
    if MARKDOWN_BOLD_HEADING.search(stripped):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    clause_lines = sum(
        1
        for line in lines
        if CLAUSE_TWO_LEVEL.match(line) or CLAUSE_THREE_LEVEL.match(line) or MARKDOWN_BOLD_HEADING.match(line)
    )
    if clause_lines >= 2:
        return True
    return len(stripped) >= GIANT_PARAGRAPH_MIN_CHARS


def _count_giant_paragraphs(nodes: list[ContentNode]) -> int:
    count = 0
    for node in nodes:
        node_type = node.type
        if node_type == "paragraph" and _is_giant_paragraph(node.text):
            count += 1
        elif node_type == "section":
            count += _count_giant_paragraphs(node.children)
    return count


def _count_tables_lists_faq(nodes: list[ContentNode]) -> tuple[int, int, int]:
    tables = lists = faq = 0
    for node in nodes:
        node_type = node.type
        if node_type == "table":
            tables += 1
        elif node_type == "list":
            lists += 1
        elif node_type == "paragraph":
            if FAQ_QUESTION.match(node.text.strip()):
                faq += 1
            for line in node.text.splitlines():
                if FAQ_QUESTION.match(line.strip()):
                    faq += 1
        elif node_type == "heading":
            if node.text.strip().endswith("?"):
                faq += 1
        elif node_type == "section":
            t, l, f = _count_tables_lists_faq(node.children)
            tables += t
            lists += l
            faq += f
    return tables, lists, faq


def _is_pdf_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if ROMAN_SECTION.match(stripped):
        return True
    if ALL_CAPS_HEADING.match(stripped) and sum(char.isalpha() for char in stripped) >= 4:
        return True
    if NUMBERED_HEADING.match(stripped):
        return True
    return False


def _split_pdf_paragraph_structure(text: str, stats: StructureRecoveryStats) -> list[ContentNode]:
    """Split PDF/OCR giant paragraphs using layout signals."""
    lines = text.splitlines()
    output: list[ContentNode] = []
    paragraph_buffer: list[str] = []
    list_buffer: list[str] = []
    current_section: SectionNode | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        body = "\n".join(paragraph_buffer).strip()
        paragraph_buffer = []
        if not body:
            return
        node = ParagraphNode(text=body)
        if current_section is not None:
            current_section.children.append(node)
        else:
            output.append(node)

    def flush_list() -> None:
        nonlocal list_buffer
        if not list_buffer:
            return
        node = ListNode(items=list_buffer, ordered=False)
        list_buffer = []
        if current_section is not None:
            current_section.children.append(node)
        else:
            output.append(node)

    def open_section(heading: str, level: int = 2) -> None:
        nonlocal current_section
        flush_list()
        flush_paragraph()
        section = SectionNode(heading=heading, level=level, children=[])
        output.append(section)
        current_section = section
        stats.recovered_sections += 1
        stats.recovered_headings += 1

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_list()
            flush_paragraph()
            continue

        bullet_match = BULLET_LINE.match(line)
        if bullet_match:
            flush_paragraph()
            list_buffer.append(bullet_match.group(1).strip())
            continue

        flush_list()

        if _is_pdf_heading_line(line):
            open_section(line)
            continue

        if CLAUSE_THREE_LEVEL.match(line) or CLAUSE_TWO_LEVEL.match(line):
            flush_paragraph()
            stats.recovered_clauses += 1
            node = ParagraphNode(text=line)
            if current_section is not None:
                current_section.children.append(node)
            else:
                output.append(node)
            continue

        paragraph_buffer.append(raw_line.rstrip())

    flush_list()
    flush_paragraph()
    if not output:
        return [ParagraphNode(text=text)]

    # Split remaining giant paragraph bodies on blank-line boundaries.
    final_output: list[ContentNode] = []
    for node in output:
        if node.type == "paragraph" and _is_giant_paragraph(node.text) and "\n\n" in node.text:
            parts = [part.strip() for part in re.split(r"\n\s*\n", node.text) if part.strip()]
            if len(parts) > 1:
                final_output.extend(ParagraphNode(text=part) for part in parts)
                continue
        if node.type == "section":
            node = SectionNode(
                heading=node.heading,
                level=node.level,
                children=_recover_nodes(node.children, stats, pdf_mode=True),
            )
        final_output.append(node)
    return final_output


def _split_paragraph_structure(text: str, stats: StructureRecoveryStats, *, pdf_mode: bool = False) -> list[ContentNode]:
    if pdf_mode:
        split = _split_pdf_paragraph_structure(text, stats)
        if len(split) > 1 or any(node.type != "paragraph" for node in split):
            return split
    lines = text.splitlines()
    output: list[ContentNode] = []
    paragraph_buffer: list[str] = []
    current_section: SectionNode | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        body = "\n".join(paragraph_buffer).strip()
        paragraph_buffer = []
        if not body:
            return
        node = ParagraphNode(text=body)
        if current_section is not None:
            current_section.children.append(node)
        else:
            output.append(node)

    def open_section(heading: str, level: int = 2) -> None:
        nonlocal current_section
        flush_paragraph()
        section = SectionNode(heading=heading, level=level, children=[])
        output.append(section)
        current_section = section
        stats.recovered_sections += 1
        stats.recovered_headings += 1

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        bold_match = MARKDOWN_BOLD_HEADING.match(line)
        if bold_match:
            inner = bold_match.group(1).strip()
            section_match = NUMBERED_SECTION_HEADING.match(inner)
            if section_match:
                open_section(f"{section_match.group(1)}. {section_match.group(2).strip()}")
                continue
            flush_paragraph()
            heading = HeadingNode(text=inner, level=3)
            stats.recovered_headings += 1
            if current_section is not None:
                current_section.children.append(heading)
            else:
                output.append(heading)
            continue

        if CLAUSE_THREE_LEVEL.match(line) or CLAUSE_TWO_LEVEL.match(line):
            flush_paragraph()
            stats.recovered_clauses += 1
            node = ParagraphNode(text=line)
            if current_section is not None:
                current_section.children.append(node)
            else:
                output.append(node)
            continue

        paragraph_buffer.append(raw_line.rstrip())

    flush_paragraph()
    return output


def _recover_nodes(nodes: list[ContentNode], stats: StructureRecoveryStats, *, pdf_mode: bool = False) -> list[ContentNode]:
    recovered: list[ContentNode] = []
    for node in nodes:
        node_type = node.type
        if node_type == "paragraph" and _is_giant_paragraph(node.text):
            split_nodes = _split_paragraph_structure(node.text, stats, pdf_mode=pdf_mode)
            recovered.extend(split_nodes)
        elif node_type == "section":
            new_section = SectionNode(
                heading=node.heading,
                level=node.level,
                children=_recover_nodes(node.children, stats, pdf_mode=pdf_mode),
            )
            recovered.append(new_section)
        else:
            recovered.append(node)
    return recovered


def _recover_page_blocks(blocks: list[ContentNode], stats: StructureRecoveryStats) -> list[ContentNode]:
    recovered = _recover_nodes(blocks, stats, pdf_mode=True)
    return _recover_nodes(recovered, stats, pdf_mode=True)


def recover_structure(content: DocumentContent) -> tuple[DocumentContent, StructureRecoveryStats]:
    """Return a structure-recovered copy of canonical content."""
    stats = StructureRecoveryStats()
    cloned = deepcopy(content)

    if cloned.pages:
        new_pages: list[PageContent] = []
        stats.giant_paragraphs_before = sum(_count_giant_paragraphs(page.blocks) for page in cloned.pages)
        for page in cloned.pages:
            new_blocks = _recover_page_blocks(page.blocks, stats)
            new_pages.append(
                PageContent(
                    page_number=page.page_number,
                    ocr_confidence=page.ocr_confidence,
                    blocks=new_blocks,
                )
            )
        cloned.pages = new_pages
        cloned.children = []
        stats.giant_paragraphs_after = sum(_count_giant_paragraphs(page.blocks) for page in cloned.pages)
    else:
        stats.giant_paragraphs_before = _count_giant_paragraphs(cloned.children)
        cloned.children = _recover_nodes(cloned.children, stats)
        stats.giant_paragraphs_after = _count_giant_paragraphs(cloned.children)

    tables, lists, faq = _count_tables_lists_faq(cloned.children)
    if cloned.pages:
        for page in cloned.pages:
            t, l, f = _count_tables_lists_faq(page.blocks)
            tables += t
            lists += l
            faq += f
    stats.faq_pairs_detected = faq
    return cloned, stats


def count_structure_elements(content: DocumentContent) -> dict[str, int]:
    """Count structural elements for quality reporting."""
    if content.pages:
        tables = lists = faq = 0
        for page in content.pages:
            t, l, f = _count_tables_lists_faq(page.blocks)
            tables += t
            lists += l
            faq += f
        giant = sum(_count_giant_paragraphs(page.blocks) for page in content.pages)
        return {
            "tables": tables,
            "lists": lists,
            "faq_structures": faq,
            "giant_paragraphs": giant,
        }

    tables, lists, faq = _count_tables_lists_faq(content.children)
    return {
        "tables": tables,
        "lists": lists,
        "faq_structures": faq,
        "giant_paragraphs": _count_giant_paragraphs(content.children),
    }
