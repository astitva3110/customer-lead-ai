"""Chunking configuration."""

from __future__ import annotations

from dataclasses import dataclass

# Dry-run default ceiling — not tied to a specific embedding model because none is configured yet.
DEFAULT_MAX_CHUNK_TOKENS = 512
DEFAULT_EMERGENCY_OVERLAP_TOKENS = 32
DEFAULT_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class ChunkingConfig:
    """Configurable chunking limits for dry-run and future production."""

    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    emergency_overlap_tokens: int = DEFAULT_EMERGENCY_OVERLAP_TOKENS
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    min_chunk_tokens: int = 8
    min_meaningful_chunk_tokens: int = 8

    def __post_init__(self) -> None:
        if self.max_chunk_tokens < 64:
            raise ValueError("max_chunk_tokens must be at least 64")
        if self.emergency_overlap_tokens < 0:
            raise ValueError("emergency_overlap_tokens must be non-negative")
