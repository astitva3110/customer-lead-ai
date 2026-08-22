"""Layout-aware PDF text extraction preserving tables and key/value specs."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from app.kb.models.structured_content import ContentNode, HeadingNode, ListNode, ParagraphNode, TableNode

LINE_Y_TOLERANCE = 4.0
COLUMN_GAP_THRESHOLD = 28.0
KEY_VALUE_MIN_ROWS = 3
KEY_VALUE_MATCH_RATIO = 0.55


@dataclass
class TextLine:
    y0: float
    x0: float
    x1: float
    text: str
    words: list[tuple]


def _group_words_into_lines(words: list[tuple], y_tolerance: float = LINE_Y_TOLERANCE) -> list[TextLine]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: (word[1], word[0]))
    lines: list[TextLine] = []
    current_words: list[tuple] = [sorted_words[0]]
    current_y = sorted_words[0][1]

    for word in sorted_words[1:]:
        if abs(word[1] - current_y) <= y_tolerance:
            current_words.append(word)
        else:
            current_words.sort(key=lambda item: item[0])
            text = " ".join(item[4] for item in current_words)
            lines.append(
                TextLine(
                    y0=min(item[1] for item in current_words),
                    x0=min(item[0] for item in current_words),
                    x1=max(item[2] for item in current_words),
                    text=text,
                    words=current_words,
                )
            )
            current_words = [word]
            current_y = word[1]

    current_words.sort(key=lambda item: item[0])
    text = " ".join(item[4] for item in current_words)
    lines.append(
        TextLine(
            y0=min(item[1] for item in current_words),
            x0=min(item[0] for item in current_words),
            x1=max(item[2] for item in current_words),
            text=text,
            words=current_words,
        )
    )
    return lines


def _split_line_columns(line: TextLine) -> list[str]:
    if not line.words:
        return [line.text.strip()] if line.text.strip() else []

    columns: list[list[str]] = [[line.words[0][4]]]
    prev_x1 = line.words[0][2]
    for word in line.words[1:]:
        if word[0] - prev_x1 >= COLUMN_GAP_THRESHOLD:
            columns.append([word[4]])
        else:
            columns[-1].append(word[4])
        prev_x1 = word[2]

    return [" ".join(column).strip() for column in columns if " ".join(column).strip()]


def _detect_key_value_table(lines: list[TextLine]) -> TableNode | None:
    rows: list[list[str]] = []
    two_column_count = 0
    for line in lines:
        columns = _split_line_columns(line)
        if len(columns) == 2:
            two_column_count += 1
            rows.append(columns)
        elif len(columns) == 1 and columns[0]:
            if ":" in columns[0]:
                label, _, value = columns[0].partition(":")
                label = label.strip()
                value = value.strip()
                if label and value:
                    two_column_count += 1
                    rows.append([label, value])

    if two_column_count < KEY_VALUE_MIN_ROWS:
        return None
    if two_column_count / max(len(lines), 1) < KEY_VALUE_MATCH_RATIO:
        return None
    return TableNode(headers=["Specification", "Value"], rows=rows)


def _detect_bullet_list(lines: list[TextLine]) -> ListNode | None:
    items: list[str] = []
    for line in lines:
        stripped = line.text.strip()
        if stripped.startswith(("- ", "• ", "● ", "▪ ", "* ")):
            items.append(stripped[2:].strip())
        elif stripped.startswith(("e ", "e\t")):
            items.append(stripped[1:].strip())
    if len(items) >= 2:
        return ListNode(ordered=False, items=items)
    return None


def _is_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 80:
        return False
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(char.isupper() for char in letters) / len(letters)
    return upper_ratio >= 0.75 and len(stripped.split()) <= 10


def _lines_to_blocks(lines: list[TextLine], page_width: float) -> list[ContentNode]:
    if not lines:
        return []

    blocks: list[ContentNode] = []
    mid_x = page_width / 2.0
    left_lines = [line for line in lines if line.x1 <= mid_x + 20]
    right_lines = [line for line in lines if line.x0 >= mid_x - 20]
    use_columns = (
        len(left_lines) >= 3
        and len(right_lines) >= 3
        and len(left_lines) + len(right_lines) >= len(lines) * 0.7
    )

    ordered_lines = lines
    if use_columns:
        left_lines.sort(key=lambda line: (line.y0, line.x0))
        right_lines.sort(key=lambda line: (line.y0, line.x0))
        ordered_lines = left_lines + [TextLine(y0=-1, x0=0, x1=0, text="", words=[])] + right_lines
        ordered_lines = [line for line in ordered_lines if line.text.strip()]

    table = _detect_key_value_table(lines)
    bullet_list = _detect_bullet_list(lines)

    if table and table.rows:
        blocks.append(table)
        table_line_texts = {" ".join(row).strip() for row in table.rows}
        remaining = [line for line in ordered_lines if line.text.strip() not in table_line_texts]
    else:
        remaining = [line for line in ordered_lines if line.text.strip()]

    if bullet_list:
        blocks.append(bullet_list)
        bullet_set = set(bullet_list.items)
        remaining = [line for line in remaining if line.text.strip() not in bullet_set]

    paragraph: list[str] = []
    for line in remaining:
        text = line.text.strip()
        if not text:
            if paragraph:
                blocks.append(ParagraphNode(text="\n".join(paragraph)))
                paragraph = []
            continue
        if _is_heading(text):
            if paragraph:
                blocks.append(ParagraphNode(text="\n".join(paragraph)))
                paragraph = []
            blocks.append(HeadingNode(text=text, level=2))
            continue
        paragraph.append(text)
    if paragraph:
        blocks.append(ParagraphNode(text="\n".join(paragraph)))

    return blocks


def _blocks_to_text(blocks: list[ContentNode]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, ParagraphNode):
            parts.append(block.text)
        elif isinstance(block, HeadingNode):
            parts.append(block.text)
        elif isinstance(block, ListNode):
            parts.extend(f"- {item}" for item in block.items)
        elif isinstance(block, TableNode):
            for row in block.rows:
                if len(row) >= 2:
                    parts.append(f"{row[0]}: {row[1]}")
                elif row:
                    parts.append(row[0])
    return "\n".join(part for part in parts if part.strip())


def extract_page_layout(page: pymupdf.Page, *, textpage: pymupdf.TextPage | None = None) -> tuple[str, list[ContentNode]]:
    words = page.get_text("words", textpage=textpage) if textpage else page.get_text("words")
    lines = _group_words_into_lines(words)
    blocks = _lines_to_blocks(lines, page.rect.width)
    if blocks:
        return _blocks_to_text(blocks), blocks

    fallback = page.get_text(textpage=textpage).strip() if textpage else page.get_text().strip()
    if fallback:
        return fallback, [ParagraphNode(text=fallback)]
    return "", []
