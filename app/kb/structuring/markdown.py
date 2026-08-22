import re

from app.kb.models.structured_content import (
    DocumentContent,
    HeadingNode,
    ListNode,
    ParagraphNode,
    SectionNode,
    TableNode,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_LIST_ITEM_RE = re.compile(r"^(\s*)[\*\-]\s+(.+)$")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")


def _parse_table(lines: list[str], start: int) -> tuple[TableNode | None, int]:
    if start >= len(lines) or not _TABLE_ROW_RE.match(lines[start].strip()):
        return None, start
    header_cells = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
    index = start + 1
    if index < len(lines) and _TABLE_SEP_RE.match(lines[index].strip()):
        index += 1
    rows: list[list[str]] = []
    while index < len(lines) and _TABLE_ROW_RE.match(lines[index].strip()):
        rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
        index += 1
    return TableNode(headers=header_cells, rows=rows), index


def _flush_paragraph(buffer: list[str], target: list) -> None:
    text = "\n".join(buffer).strip()
    buffer.clear()
    if text:
        target.append(ParagraphNode(text=text))


def markdown_to_structured(title: str, markdown: str) -> DocumentContent:
    lines = markdown.split("\n")
    children: list = []
    current_section: SectionNode | None = None
    paragraph_buffer: list[str] = []
    index = 0

    def append_node(node) -> None:
        nonlocal current_section
        if current_section is not None:
            current_section.children.append(node)
        else:
            children.append(node)

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            _flush_paragraph(paragraph_buffer, current_section.children if current_section else children)
            index += 1
            continue

        table, next_index = _parse_table(lines, index)
        if table is not None:
            _flush_paragraph(paragraph_buffer, current_section.children if current_section else children)
            append_node(table)
            index = next_index
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match and "![" not in stripped:
            _flush_paragraph(paragraph_buffer, current_section.children if current_section else children)
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level <= 2:
                current_section = SectionNode(heading=heading_text, level=level, children=[])
                children.append(current_section)
            else:
                append_node(HeadingNode(text=heading_text, level=level))
            index += 1
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            _flush_paragraph(paragraph_buffer, current_section.children if current_section else children)
            items: list[str] = [list_match.group(2).strip()]
            index += 1
            while index < len(lines):
                next_match = _LIST_ITEM_RE.match(lines[index])
                if not next_match or len(next_match.group(1)) != len(list_match.group(1)):
                    break
                items.append(next_match.group(2).strip())
                index += 1
            append_node(ListNode(items=items))
            continue

        paragraph_buffer.append(stripped)
        index += 1

    _flush_paragraph(paragraph_buffer, current_section.children if current_section else children)

    if not children and markdown.strip():
        children.append(ParagraphNode(text=markdown.strip()))

    return DocumentContent(type="document", title=title, children=children)
