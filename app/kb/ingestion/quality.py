"""Chunk quality gate for Phase 12 document ingestion."""

from __future__ import annotations

import re
import statistics
from typing import Any

from app.config import settings
from app.kb.ingestion.models import Phase12ChunkRecord

MAX_TOKENS = 512
OCR_GARBAGE_RE = re.compile(r"^[^a-zA-Z0-9]{0,5}$")
UI_DEBRIS = (
    "page ",
    "you may also like",
    "add to cart",
    "write a review",
    "sort by",
)
FORBIDDEN_METADATA = ("embedding_version:", "chunk_id:", "document_version:")


def _is_ocr_garbage(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 4 and OCR_GARBAGE_RE.match(stripped):
        return True
    alpha_ratio = sum(char.isalpha() for char in stripped) / max(len(stripped), 1)
    return alpha_ratio < 0.15 and len(stripped) > 20


def _is_ui_debris(text: str) -> bool:
    lowered = text.lower().strip()
    if len(lowered) < 120:
        return any(token in lowered for token in UI_DEBRIS)
    return False


def validate_chunk(chunk: Phase12ChunkRecord) -> list[str]:
    issues: list[str] = []
    if not chunk.content.strip():
        issues.append("empty_content")
    if chunk.token_count <= 0:
        issues.append("invalid_token_count")
    if chunk.token_count > MAX_TOKENS:
        issues.append("token_limit_exceeded")
    if not chunk.chunk_id:
        issues.append("missing_chunk_id")
    if not chunk.document_id:
        issues.append("missing_document_id")
    if not chunk.source_file_hash:
        issues.append("missing_source_file_hash")
    if not chunk.embedding_input:
        issues.append("missing_embedding_input")
    if not chunk.embedding_input_hash:
        issues.append("missing_embedding_input_hash")
    for token in FORBIDDEN_METADATA:
        if token in chunk.content.lower():
            issues.append("forbidden_metadata_leakage")
    if _is_ocr_garbage(chunk.content):
        issues.append("ocr_garbage")
    if _is_ui_debris(chunk.content):
        issues.append("ui_debris")
    return issues


class ChunkQualityGate:
    def evaluate(self, chunks: list[Phase12ChunkRecord]) -> dict[str, Any]:
        passed: list[Phase12ChunkRecord] = []
        failed: list[dict[str, Any]] = []
        heading_only = 0
        small_chunks = 0
        table_chunks = 0
        token_counts: list[int] = []

        for chunk in chunks:
            issues = validate_chunk(chunk)
            token_counts.append(chunk.token_count)
            if chunk.content_type == "heading":
                heading_only += 1
            if chunk.token_count < 25:
                small_chunks += 1
            if chunk.content_type == "table":
                table_chunks += 1
            if issues:
                failed.append({"chunk_id": chunk.chunk_id, "issues": issues, "content_preview": chunk.content[:120]})
            else:
                passed.append(chunk)

        max_tokens = max(token_counts) if token_counts else 0
        median_tokens = statistics.median(token_counts) if token_counts else 0
        return {
            "passed": passed,
            "failed": failed,
            "statistics": {
                "chunks_generated": len(chunks),
                "chunks_passed": len(passed),
                "chunks_failed": len(failed),
                "heading_only_chunks": heading_only,
                "small_chunks": small_chunks,
                "table_chunks": table_chunks,
                "max_token_count": max_tokens,
                "median_token_count": median_tokens,
                "max_tokens_within_limit": max_tokens <= MAX_TOKENS,
                "chunking_algorithm_version": settings.phase12_chunking_algorithm_version,
            },
            "determinism_status": "PASS" if not failed else "FAIL",
        }


def validate_chunk_set(
    chunks: list[Phase12ChunkRecord],
    passed: list[Phase12ChunkRecord],
) -> list[str]:
    """Hard-gate checks for the chunk set that would be embedded."""
    issues: list[str] = []
    if not passed:
        issues.append("no_valid_chunks")
        return issues
    chunk_ids = [chunk.chunk_id for chunk in passed]
    if len(chunk_ids) != len(set(chunk_ids)):
        issues.append("duplicate_chunk_ids")
    hashes = [chunk.embedding_input_hash for chunk in passed]
    if len(hashes) != len(set(hashes)):
        issues.append("duplicate_chunks")
    for chunk in passed:
        if not chunk.document_id:
            issues.append("missing_document_id")
        if not chunk.embedding_input.strip():
            issues.append("invalid_embedding_input")
        if chunk.page_number is None and not chunk.source_file_hash:
            issues.append("missing_source_information")
    return list(dict.fromkeys(issues))
