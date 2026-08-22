"""Build hierarchical semantic units from structured retrieval content."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.kb.enums import DocumentType
from app.kb.models.structured_content import (
    ContentNode,
    DocumentContent,
    HeadingNode,
    ListNode,
    ParagraphNode,
    SectionNode,
    TableNode,
)

CLAUSE_PATTERN = re.compile(r"^(\d+(?:\.\d+)+)\s+")
FAQ_QUESTION = re.compile(r"^Q(\d+)\.\s+(.+)$", re.I)
PARENT_SECTION_LABEL = re.compile(r"^\d+\.\s+[A-Za-z]")


@dataclass
class SemanticUnit:
    """Coherent knowledge unit before token-limited chunk materialization."""

    content: str
    section_path: list[str] = field(default_factory=list)
    page_number: int | None = None
    unit_type: str = "semantic"
    children: list[SemanticUnit] = field(default_factory=list)
    source_block_path: list[str] = field(default_factory=list)


def _clean_title(title: str) -> str:
    return " ".join(title.split())


def _is_prospectus(title: str, url: str) -> bool:
    return "prospectus" in f"{title} {url}".lower() and "/investor/ipo/" in url.lower()


def effective_document_type(document_type: DocumentType, title: str, url: str) -> str:
    if _is_prospectus(title, url):
        return "prospectus"
    if document_type == DocumentType.FAQ:
        return "faq"
    return document_type.value


def _render_table(table: TableNode) -> str:
    lines: list[str] = []
    if table.headers:
        lines.append(" | ".join(table.headers))
    for row in table.rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _render_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item.strip()}" for item in items if item.strip())


def _append_path(path: list[str], label: str | None) -> list[str]:
    if not label or not label.strip():
        return list(path)
    cleaned = label.strip()
    if path and path[-1] == cleaned:
        return list(path)
    return [*path, cleaned]


def _nodes_from_content(content: DocumentContent) -> list[tuple[list[ContentNode], int | None]]:
    if content.pages:
        return [(page.blocks, page.page_number) for page in content.pages]
    return [(content.children, None)]


def _collect_faq_pair(nodes: list[ContentNode], index: int) -> tuple[SemanticUnit | None, int]:
    node = nodes[index]
    if node.type != "heading" and node.type != "paragraph":
        return None, index
    question = node.text.strip() if node.type == "heading" else node.text.strip()
    if node.type == "paragraph" and not FAQ_QUESTION.match(question) and not question.endswith("?"):
        return None, index
    if index + 1 >= len(nodes):
        return None, index
    nxt = nodes[index + 1]
    if nxt.type != "paragraph":
        return None, index
    answer = nxt.text.strip()
    unit = SemanticUnit(
        content=f"Question: {question}\n\nAnswer: {answer}",
        unit_type="faq_pair",
    )
    return unit, index + 2


def _unit_from_nodes(
    nodes: list[ContentNode],
    *,
    section_path: list[str],
    page_number: int | None,
    block_prefix: list[str],
    document_type: str,
) -> list[SemanticUnit]:
    units: list[SemanticUnit] = []
    index = 0
    while index < len(nodes):
        if document_type == "faq":
            faq_unit, next_index = _collect_faq_pair(nodes, index)
            if faq_unit is not None:
                faq_unit.section_path = list(section_path)
                faq_unit.page_number = page_number
                faq_unit.source_block_path = [*block_prefix, f"faq:{index}"]
                units.append(faq_unit)
                index = next_index
                continue

        node = nodes[index]
        node_type = node.type
        path = [*block_prefix, f"{node_type}:{index}"]

        if node_type == "section":
            heading = (node.heading or "").strip()
            child_path = _append_path(section_path, heading)
            child_units = _unit_from_nodes(
                node.children,
                section_path=child_path,
                page_number=page_number,
                block_prefix=[*path, "children"],
                document_type=document_type,
            )
            if child_units:
                units.append(
                    SemanticUnit(
                        content="",
                        section_path=child_path,
                        page_number=page_number,
                        unit_type="section",
                        children=child_units,
                        source_block_path=path,
                    )
                )
            index += 1
            continue

        if node_type == "heading":
            heading = node.text.strip()
            child_path = _append_path(section_path, heading)
            # Look ahead for FAQ-style heading + answer paragraph.
            if document_type == "faq" and index + 1 < len(nodes) and nodes[index + 1].type == "paragraph":
                answer = nodes[index + 1].text.strip()
                units.append(
                    SemanticUnit(
                        content=f"Question: {heading}\n\nAnswer: {answer}",
                        section_path=child_path,
                        page_number=page_number,
                        unit_type="faq_pair",
                        source_block_path=path,
                    )
                )
                index += 2
                continue
            units.append(
                SemanticUnit(
                    content=heading,
                    section_path=child_path,
                    page_number=page_number,
                    unit_type="heading",
                    source_block_path=path,
                )
            )
            index += 1
            continue

        if node_type == "paragraph":
            text = node.text.strip()
            if not text:
                index += 1
                continue
            unit_type = "clause" if CLAUSE_PATTERN.match(text) else "paragraph"
            units.append(
                SemanticUnit(
                    content=text,
                    section_path=list(section_path),
                    page_number=page_number,
                    unit_type=unit_type,
                    source_block_path=path,
                )
            )
            index += 1
            continue

        if node_type == "list":
            text = _render_list(node.items)
            if text.strip():
                units.append(
                    SemanticUnit(
                        content=text,
                        section_path=list(section_path),
                        page_number=page_number,
                        unit_type="list",
                        source_block_path=path,
                    )
                )
            index += 1
            continue

        if node_type == "table":
            text = _render_table(node)
            if text.strip():
                units.append(
                    SemanticUnit(
                        content=text,
                        section_path=list(section_path),
                        page_number=page_number,
                        unit_type="table",
                        source_block_path=path,
                    )
                )
            index += 1
            continue

        index += 1

    return units


def _is_parent_section_label(text: str) -> bool:
    """Top-level numbered section title such as '2. Eligibility' (not '2.1 ...')."""
    stripped = text.strip()
    if not stripped or re.match(r"^\d+\.\d+", stripped):
        return False
    return bool(PARENT_SECTION_LABEL.match(stripped)) and len(stripped.split()) <= 10


def _first_meaningful_unit(unit: SemanticUnit) -> SemanticUnit | None:
    if unit.content.strip():
        return unit
    for child in unit.children:
        found = _first_meaningful_unit(child)
        if found is not None:
            return found
    return None


def _attach_parent_label_to_unit(parent: SemanticUnit, target: SemanticUnit) -> SemanticUnit:
    parent_label = parent.content.strip()
    combined = f"{parent_label}\n\n{target.content.strip()}"
    section_path = _append_path(parent.section_path, parent_label)
    if target.section_path:
        last = target.section_path[-1]
        if last != parent_label:
            section_path = _append_path(section_path, last)
    return SemanticUnit(
        content=combined,
        section_path=section_path,
        page_number=target.page_number,
        unit_type=target.unit_type,
        children=target.children,
        source_block_path=[*parent.source_block_path, "parent_heading_attached", *target.source_block_path],
    )


def _merge_heading_with_following(units: list[SemanticUnit]) -> list[SemanticUnit]:
    """Attach standalone headings/parent section labels to the next semantic unit."""
    if not units:
        return units

    merged: list[SemanticUnit] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if unit.children:
            unit.children = _merge_heading_with_following(unit.children)
            merged.append(unit)
            index += 1
            continue

        is_parent_label = unit.unit_type in {"heading", "paragraph", "clause"} and _is_parent_section_label(
            unit.content
        )
        is_standalone_heading = unit.unit_type == "heading"

        if index + 1 < len(units):
            follower = units[index + 1]

            if (is_standalone_heading or is_parent_label) and follower.unit_type == "section":
                meaningful = _first_meaningful_unit(follower)
                if meaningful is not None:
                    attached = _attach_parent_label_to_unit(unit, meaningful)
                    meaningful.content = attached.content
                    meaningful.section_path = attached.section_path
                    meaningful.source_block_path = attached.source_block_path
                    merged.append(follower)
                    index += 2
                    continue

            if (
                is_standalone_heading
                and not follower.children
                and follower.unit_type in {"paragraph", "clause", "list", "table"}
            ):
                combined = f"{unit.content.strip()}\n\n{follower.content.strip()}"
                merged.append(
                    SemanticUnit(
                        content=combined,
                        section_path=follower.section_path,
                        page_number=follower.page_number,
                        unit_type=follower.unit_type,
                        source_block_path=[*unit.source_block_path, "heading_attached", *follower.source_block_path],
                    )
                )
                index += 2
                continue

            if is_parent_label and not follower.children and follower.unit_type in {
                "paragraph",
                "clause",
                "list",
                "table",
            }:
                combined = f"{unit.content.strip()}\n\n{follower.content.strip()}"
                merged.append(
                    SemanticUnit(
                        content=combined,
                        section_path=follower.section_path,
                        page_number=follower.page_number,
                        unit_type=follower.unit_type,
                        source_block_path=[*unit.source_block_path, "parent_heading_attached", *follower.source_block_path],
                    )
                )
                index += 2
                continue

        merged.append(unit)
        index += 1

    return merged


CLAUSE_PREFIX = re.compile(r"^\d+(?:\.\d+)+\s+")
ADDRESS_HINT = re.compile(r"(limited|noida|sector|india|floor|building|gautam|uttar pradesh)", re.I)


def _merge_clause_with_address_followers(units: list[SemanticUnit]) -> list[SemanticUnit]:
    """Attach address paragraphs that follow clause lines ending with ':'."""
    if not units:
        return units

    merged: list[SemanticUnit] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if unit.children:
            unit.children = _merge_clause_with_address_followers(unit.children)
            merged.append(unit)
            index += 1
            continue

        if (
            unit.unit_type in {"clause", "paragraph"}
            and unit.content.rstrip().endswith(":")
            and index + 1 < len(units)
        ):
            followers: list[SemanticUnit] = []
            follower_index = index + 1
            while follower_index < len(units):
                follower = units[follower_index]
                if follower.unit_type != "paragraph" or CLAUSE_PREFIX.match(follower.content.strip()):
                    break
                if followers or ADDRESS_HINT.search(follower.content):
                    followers.append(follower)
                    follower_index += 1
                    continue
                break

            if followers:
                combined = unit.content.strip()
                for follower in followers:
                    combined = f"{combined}\n\n{follower.content.strip()}"
                merged.append(
                    SemanticUnit(
                        content=combined,
                        section_path=unit.section_path,
                        page_number=unit.page_number,
                        unit_type=unit.unit_type,
                        source_block_path=[
                            *unit.source_block_path,
                            "address_attached",
                            *[f.source_block_path[-1] for f in followers],
                        ],
                    )
                )
                index = follower_index
                continue

        merged.append(unit)
        index += 1

    return merged


def build_semantic_units(
    *,
    structured_content: DocumentContent,
    document_type: DocumentType,
    title: str,
    canonical_url: str,
) -> list[SemanticUnit]:
    """Create top-level semantic units from structured content."""
    doc_type = effective_document_type(document_type, title, canonical_url)
    root_path = [_clean_title(title)] if title.strip() else []
    all_units: list[SemanticUnit] = []

    for blocks, page_number in _nodes_from_content(structured_content):
        page_units = _unit_from_nodes(
            blocks,
            section_path=root_path,
            page_number=page_number,
            block_prefix=[f"page:{page_number}"] if page_number is not None else ["root"],
            document_type=doc_type,
        )
        if page_number is not None and page_units:
            all_units.append(
                SemanticUnit(
                    content="",
                    section_path=root_path,
                    page_number=page_number,
                    unit_type="page",
                    children=page_units,
                    source_block_path=[f"page:{page_number}"],
                )
            )
        else:
            all_units.extend(page_units)

    return _merge_clause_with_address_followers(_merge_heading_with_following(all_units))
