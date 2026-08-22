"""Normalize extracted units into hierarchical structured content."""

from __future__ import annotations

from app.kb.ingestion.models import ContentUnitType, ExtractionResult
from app.kb.models.structured_content import (
    DocumentContent,
    HeadingNode,
    ListNode,
    PageContent,
    ParagraphNode,
    SectionNode,
    TableNode,
)
from app.kb.retrieval.models import StructureRecoveryStats
from app.kb.retrieval.recovery import recover_structure


def _unit_to_node(unit) -> HeadingNode | ParagraphNode | ListNode | TableNode:
    if unit.content_type == ContentUnitType.HEADING:
        return HeadingNode(text=unit.text, level=unit.heading_level or 2)
    if unit.content_type == ContentUnitType.LIST:
        return ListNode(items=unit.list_items or [unit.text], ordered=unit.list_ordered)
    if unit.content_type == ContentUnitType.TABLE:
        return TableNode(headers=unit.table_headers, rows=unit.table_rows)
    return ParagraphNode(text=unit.text)


def build_structured_document(extraction: ExtractionResult) -> tuple[DocumentContent, StructureRecoveryStats]:
    """Build section hierarchy from extracted units — primary chunking input."""
    pages: dict[int, list] = {}
    root_nodes: list = []
    current_section: SectionNode | None = None
    current_page: int | None = None

    def flush_section_to_page() -> None:
        nonlocal current_section
        if current_section is None:
            return
        if current_page is not None:
            pages.setdefault(current_page, []).append(current_section)
        else:
            root_nodes.append(current_section)
        current_section = None

    for unit in extraction.units:
        if unit.page_number != current_page:
            flush_section_to_page()
            current_page = unit.page_number

        node = _unit_to_node(unit)
        if unit.content_type == ContentUnitType.HEADING:
            flush_section_to_page()
            current_section = SectionNode(heading=unit.text, level=unit.heading_level or 2, children=[])
            continue

        if current_section is not None:
            current_section.children.append(node)
        elif current_page is not None:
            pages.setdefault(current_page, []).append(node)
        else:
            root_nodes.append(node)

    flush_section_to_page()

    page_models = [
        PageContent(page_number=page_number, blocks=blocks)
        for page_number, blocks in sorted(pages.items())
    ] or None

    content = DocumentContent(
        type="document",
        title=extraction.title,
        children=root_nodes,
        pages=page_models,
    )
    return recover_structure(content)


def structured_to_retrieval_text(content: DocumentContent) -> str:
    """Compact hierarchical preview text — not used as primary chunk source."""
    parts: list[str] = []

    def walk(nodes) -> None:
        for node in nodes:
            node_type = node.type
            if node_type == "section":
                if node.heading:
                    parts.append(str(node.heading))
                walk(node.children or [])
            elif node_type == "heading":
                parts.append(str(node.text))
            elif node_type == "paragraph":
                parts.append(str(node.text))
            elif node_type == "list":
                parts.extend(str(item) for item in node.items or [])
            elif node_type == "table":
                if node.headers:
                    parts.append(" | ".join(node.headers))
                for row in node.rows or []:
                    parts.append(" | ".join(row))

    if content.pages:
        for page in content.pages:
            walk(page.blocks)
    walk(content.children)
    return "\n\n".join(part for part in parts if part.strip())
