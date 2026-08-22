"""Plain text extraction and consistency checks."""

from __future__ import annotations

import re

from app.kb.models.canonical import CanonicalDocument
from app.kb.models.structured_content import DocumentContent

_MEANINGFUL_RE = re.compile(r"[\w]", re.UNICODE)


def plain_text_from_structured(structured: DocumentContent) -> str:
    parts: list[str] = []

    def walk(nodes) -> None:
        for node in nodes:
            node_type = getattr(node, "type", None)
            if node_type == "paragraph":
                parts.append(node.text)
            elif node_type == "heading":
                parts.append(node.text)
            elif node_type == "list":
                parts.extend(node.items)
            elif node_type == "table":
                if node.headers:
                    parts.append(" | ".join(node.headers))
                for row in node.rows:
                    parts.append(" | ".join(row))
            elif node_type == "section":
                if node.heading:
                    parts.append(node.heading)
                walk(node.children)

    if structured.pages:
        for page in structured.pages:
            walk(page.blocks)
    else:
        walk(structured.children)
    return "\n\n".join(part for part in parts if part.strip())


def _normalize_for_compare(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _meaningful_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[\w]{3,}", text, re.UNICODE)}


def check_content_consistency(document: CanonicalDocument) -> list[str]:
    """Compare plain_text against structured_content (approximate, not exact)."""
    issues: list[str] = []
    structured = document.structured_content
    if structured is None:
        return ["missing structured_content"]

    derived = plain_text_from_structured(structured)
    plain = document.plain_text or ""

    if not plain.strip():
        issues.append("plain_text is empty")
        return issues

    if not derived.strip() and not structured.pages and not structured.children:
        issues.append("structured_content is empty but plain_text is not")
        return issues

    # For PDFs plain_text often equals cleaned raw text; structure may be partial
    if document.source_type.value in {"pdf", "scanned_pdf"}:
        plain_meaningful = len(_MEANINGFUL_RE.findall(plain))
        derived_meaningful = len(_MEANINGFUL_RE.findall(derived))
        if derived_meaningful > 0 and plain_meaningful > 0:
            ratio = min(derived_meaningful, plain_meaningful) / max(derived_meaningful, plain_meaningful)
            if ratio < 0.3:
                issues.append(
                    f"pdf plain_text vs structured meaningful char ratio very low ({ratio:.0%})"
                )
        return issues

    plain_tokens = _meaningful_tokens(_normalize_for_compare(plain))
    derived_tokens = _meaningful_tokens(_normalize_for_compare(derived))

    if not derived_tokens:
        if plain_tokens:
            issues.append("structured_content missing tokens present in plain_text")
        return issues

    overlap = plain_tokens & derived_tokens
    if plain_tokens:
        coverage = len(overlap) / len(plain_tokens)
        if coverage < 0.5:
            issues.append(f"plain_text token coverage in structure is low ({coverage:.0%})")
    if derived_tokens - plain_tokens and len(derived_tokens - plain_tokens) > len(plain_tokens) * 0.5:
        issues.append("structured_content contains many tokens absent from plain_text")

    return issues


def find_empty_nodes(structured: DocumentContent) -> list[str]:
    issues: list[str] = []

    def walk(nodes, path: str) -> None:
        for index, node in enumerate(nodes):
            node_type = getattr(node, "type", None)
            loc = f"{path}[{index}]/{node_type}"
            if node_type == "paragraph" and not node.text.strip():
                issues.append(f"empty paragraph at {loc}")
            elif node_type == "list" and not node.items:
                issues.append(f"empty list at {loc}")
            elif node_type == "table" and not node.headers and not node.rows:
                issues.append(f"empty table at {loc}")
            elif node_type == "section":
                if not node.heading and not node.children:
                    issues.append(f"empty section at {loc}")
                walk(node.children, loc)

    if structured.pages:
        for page in structured.pages:
            if not page.blocks:
                issues.append(f"empty pdf page {page.page_number}")
            walk(page.blocks, f"page{page.page_number}")
    else:
        walk(structured.children, "root")
    return issues
