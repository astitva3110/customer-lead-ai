"""Cross-document duplicate suppression for V3 — prefer canonical policy doc."""

from __future__ import annotations

import re

from app.kb.ingestion.models import Phase12ChunkRecord

WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text.strip().lower())


def _content_key(chunk: Phase12ChunkRecord) -> str:
    normalized = _normalize(chunk.content)
    path = "|".join(chunk.section_path)
    return f"{chunk.document_id}:{path}:{normalized}"


def suppress_cross_document_duplicates(
    documents: list[tuple[str, list[Phase12ChunkRecord]]],
) -> tuple[list[Phase12ChunkRecord], list[dict]]:
    """Prefer terms (policy) over merged when duplicate knowledge exists."""
    priority = {"terms": 0, "merged": 1}
    kept: list[Phase12ChunkRecord] = []
    suppressed: list[dict] = []
    seen: dict[str, tuple[str, Phase12ChunkRecord]] = {}

    ordered = sorted(documents, key=lambda item: priority.get(item[0], 99))
    for label, chunks in ordered:
        for chunk in chunks:
            key = _content_key(chunk)
            norm = _normalize(chunk.content)
            if len(norm) < 40:
                kept.append(chunk)
                continue
            existing = seen.get(key)
            if existing is None:
                seen[key] = (label, chunk)
                kept.append(chunk)
                continue
            existing_label, existing_chunk = existing
            if priority.get(label, 99) < priority.get(existing_label, 99):
                kept = [item for item in kept if item.chunk_id != existing_chunk.chunk_id]
                suppressed.append(
                    {
                        "chunk_id": existing_chunk.chunk_id,
                        "document": existing_label,
                        "reason": "duplicate_canonical_preferred",
                        "kept_chunk_id": chunk.chunk_id,
                        "kept_document": label,
                    }
                )
                seen[key] = (label, chunk)
                kept.append(chunk)
            else:
                suppressed.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document": label,
                        "reason": "duplicate_canonical_preferred",
                        "kept_chunk_id": existing_chunk.chunk_id,
                        "kept_document": existing_label,
                    }
                )
    return kept, suppressed
