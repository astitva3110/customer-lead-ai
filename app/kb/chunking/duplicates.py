"""Context-aware duplicate chunk classification and suppression."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.kb.chunking.models import ChunkRecord

ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9\s&,\-./():'\"]{2,80}$")
BOILERPLATE = re.compile(
    r"(earkart limited|www\.earkart|left blank|by order of the board|make in india|all rights reserved)",
    re.I,
)


class DuplicateClassification(StrEnum):
    SAME_CONTEXT_DUPLICATE = "SAME_CONTEXT_DUPLICATE"
    DIFFERENT_CONTEXT_REPEAT = "DIFFERENT_CONTEXT_REPEAT"
    DUPLICATE_HEADING = "DUPLICATE_HEADING"
    BOILERPLATE = "BOILERPLATE"
    FALSE_POSITIVE = "FALSE_POSITIVE"


@dataclass(frozen=True)
class DuplicateEntry:
    chunk: ChunkRecord
    classification: DuplicateClassification
    duplicate_of_chunk_id: str | None = None


def _normalize_content(text: str) -> str:
    return " ".join(text.split()).lower()


def _body_text(content: str) -> str:
    if "\n\n" in content:
        return content.split("\n\n")[-1].strip()
    return content.strip()


def duplicate_content_key(chunk: ChunkRecord) -> str:
    normalized = _normalize_content(chunk.content)
    path = "|".join(chunk.section_path)
    return f"{chunk.document_id}:{path}:{normalized}"


def duplicate_content_key_relaxed(chunk: ChunkRecord) -> str:
    """Content-only key for cross-context repeat detection."""
    return f"{chunk.document_id}:{_normalize_content(chunk.content)}"


def classify_duplicate_pair(first: ChunkRecord, duplicate: ChunkRecord) -> DuplicateClassification:
    body = _body_text(duplicate.content)
    same_path = tuple(first.section_path) == tuple(duplicate.section_path)

    if same_path:
        if BOILERPLATE.search(duplicate.content) or "left blank" in body.lower():
            return DuplicateClassification.BOILERPLATE
        return DuplicateClassification.SAME_CONTEXT_DUPLICATE

    if BOILERPLATE.search(duplicate.content) or "left blank" in body.lower():
        return DuplicateClassification.BOILERPLATE

    if len(body) >= 40 or "." in body or ":" in body:
        return DuplicateClassification.DIFFERENT_CONTEXT_REPEAT

    if ALL_CAPS.match(body) or len(body) < 35:
        return DuplicateClassification.DUPLICATE_HEADING

    if "you may also like" in duplicate.content.lower():
        return DuplicateClassification.DUPLICATE_HEADING

    return DuplicateClassification.DIFFERENT_CONTEXT_REPEAT


def suppress_same_context_duplicates(
    chunks: list[ChunkRecord],
) -> tuple[list[ChunkRecord], list[DuplicateEntry]]:
    """Suppress only same-context duplicates and obvious boilerplate repeats."""
    seen_strict: dict[str, ChunkRecord] = {}
    kept: list[ChunkRecord] = []
    suppressed: list[DuplicateEntry] = []

    for chunk in chunks:
        strict_key = duplicate_content_key(chunk)
        if strict_key in seen_strict:
            first = seen_strict[strict_key]
            classification = classify_duplicate_pair(first, chunk)
            if classification in {
                DuplicateClassification.SAME_CONTEXT_DUPLICATE,
                DuplicateClassification.BOILERPLATE,
            }:
                suppressed.append(
                    DuplicateEntry(
                        chunk=chunk,
                        classification=classification,
                        duplicate_of_chunk_id=first.chunk_id,
                    )
                )
                continue
        seen_strict[strict_key] = chunk
        kept.append(chunk)

    return kept, suppressed


def analyze_document_duplicates(chunks: list[ChunkRecord]) -> list[DuplicateEntry]:
    """Classify all duplicate occurrences within a document for reporting."""
    seen_relaxed: dict[str, ChunkRecord] = {}
    entries: list[DuplicateEntry] = []

    for chunk in chunks:
        relaxed_key = duplicate_content_key_relaxed(chunk)
        if relaxed_key in seen_relaxed:
            first = seen_relaxed[relaxed_key]
            entries.append(
                DuplicateEntry(
                    chunk=chunk,
                    classification=classify_duplicate_pair(first, chunk),
                    duplicate_of_chunk_id=first.chunk_id,
                )
            )
        else:
            seen_relaxed[relaxed_key] = chunk

    return entries
