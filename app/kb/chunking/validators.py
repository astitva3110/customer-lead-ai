"""Chunk validation for dry-run quality reporting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.duplicates import duplicate_content_key
from app.kb.chunking.models import ChunkRecord, SplitMethod

PIPELINE_METADATA = re.compile(r"sha256:[a-f0-9]{64}", re.I)


@dataclass
class ChunkValidationIssue:
    severity: str
    category: str
    message: str
    document_id: str
    chunk_id: str | None = None
    evidence: str | None = None


@dataclass
class ChunkValidationResult:
    valid: bool
    issues: list[ChunkValidationIssue] = field(default_factory=list)


def validate_chunks(chunks: list[ChunkRecord], config: ChunkingConfig) -> ChunkValidationResult:
    issues: list[ChunkValidationIssue] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()

    for chunk in chunks:
        if not chunk.content.strip():
            issues.append(
                ChunkValidationIssue(
                    severity="P1",
                    category="empty_content",
                    message="Chunk content is empty",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if chunk.token_count <= 0:
            issues.append(
                ChunkValidationIssue(
                    severity="P1",
                    category="invalid_token_count",
                    message="Chunk token count is invalid",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if (
            chunk.token_count > config.max_chunk_tokens
            and chunk.split_method != SplitMethod.HARD_TOKEN_FALLBACK
        ):
            issues.append(
                ChunkValidationIssue(
                    severity="P1",
                    category="token_limit_exceeded",
                    message="Chunk exceeds configured maximum without emergency fallback",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    evidence=f"tokens={chunk.token_count}, max={config.max_chunk_tokens}",
                )
            )
        if not chunk.document_id:
            issues.append(
                ChunkValidationIssue(
                    severity="P0",
                    category="missing_document_id",
                    message="Chunk missing document_id",
                    document_id="",
                    chunk_id=chunk.chunk_id,
                )
            )
        if not chunk.source_url or not chunk.canonical_url:
            issues.append(
                ChunkValidationIssue(
                    severity="P1",
                    category="missing_provenance_url",
                    message="Chunk missing source or canonical URL",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        if chunk.chunk_id in seen_ids:
            issues.append(
                ChunkValidationIssue(
                    severity="P0",
                    category="duplicate_chunk_id",
                    message="Duplicate chunk_id within document batch",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        seen_ids.add(chunk.chunk_id)

        content_key = duplicate_content_key(chunk)
        if content_key in seen_content:
            issues.append(
                ChunkValidationIssue(
                    severity="P2",
                    category="duplicate_chunk_content",
                    message="Exact duplicate chunk content within document and section path",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )
        seen_content.add(content_key)

        if PIPELINE_METADATA.search(chunk.content):
            issues.append(
                ChunkValidationIssue(
                    severity="P1",
                    category="pipeline_metadata_leak",
                    message="Pipeline metadata detected in chunk content",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )

        if chunk.overlap_tokens > 0 and chunk.split_method not in {
            SplitMethod.HARD_TOKEN_FALLBACK,
            SplitMethod.SENTENCE,
        }:
            issues.append(
                ChunkValidationIssue(
                    severity="P2",
                    category="unexpected_overlap",
                    message="Non-emergency chunk has overlap tokens",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    evidence=f"overlap_tokens={chunk.overlap_tokens}",
                )
            )

        if chunk.token_count < config.min_chunk_tokens and chunk.split_method in {
            SplitMethod.SEMANTIC,
            SplitMethod.HEADING,
        }:
            issues.append(
                ChunkValidationIssue(
                    severity="P3",
                    category="very_small_chunk",
                    message="Very small semantic chunk",
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    evidence=f"tokens={chunk.token_count}",
                )
            )

    blocking = [issue for issue in issues if issue.severity in {"P0", "P1"}]
    return ChunkValidationResult(valid=not blocking, issues=issues)
