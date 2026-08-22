"""Build concise catalog summaries from existing V3.1 corpus chunks."""

from __future__ import annotations

from app.kb.ingestion.models import Phase12ChunkRecord

PRODUCT_SECTION_MARKERS = ("6.1 tiny", "6.2 bluup", "6.3 bluup+")
HEARING_AID_TYPE = "hearing_aid_type"


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        text = line.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _is_product_atom(chunk: Phase12ChunkRecord) -> bool:
    path = " > ".join(chunk.section_path).lower()
    if chunk.content_type in {"product_specification", "product_description"}:
        return True
    return any(marker in path for marker in PRODUCT_SECTION_MARKERS)


def _is_hearing_aid_type_atom(chunk: Phase12ChunkRecord) -> bool:
    return chunk.content_type == HEARING_AID_TYPE


def build_product_catalog_summary_content(chunks: list[Phase12ChunkRecord]) -> str:
    product_atoms = [chunk for chunk in chunks if _is_product_atom(chunk)]
    lines = ["Earkart Product Catalog Summary"]
    for chunk in sorted(product_atoms, key=lambda item: item.section_path):
        section = chunk.section_path[-1] if chunk.section_path else "Product"
        body = chunk.content.strip().replace("\n", " ")
        if len(body) > 220:
            body = body[:217].rstrip() + "..."
        lines.append(f"{section}: {body}")
    return "\n".join(_dedupe_lines(lines)).strip()


def build_hearing_aid_catalog_summary_content(chunks: list[Phase12ChunkRecord]) -> str:
    type_atoms = [chunk for chunk in chunks if _is_hearing_aid_type_atom(chunk)]
    lines = ["Earkart Hearing Aid Catalog Summary"]
    for chunk in sorted(type_atoms, key=lambda item: item.section_path):
        section = chunk.section_path[-1] if chunk.section_path else "Hearing Aid Type"
        body = chunk.content.strip().replace("\n", " ")
        if len(body) > 220:
            body = body[:217].rstrip() + "..."
        lines.append(f"{section}: {body}")
    return "\n".join(_dedupe_lines(lines)).strip()
