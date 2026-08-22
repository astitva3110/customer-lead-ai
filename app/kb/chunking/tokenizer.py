"""Configurable tokenizer abstraction for chunk size estimation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.kb.chunking.config import DEFAULT_CHARS_PER_TOKEN


class Tokenizer(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        """Return approximate token count for text."""

    @abstractmethod
    def encode_length(self, text: str) -> int:
        """Alias for count — used by splitters."""


class CharacterEstimateTokenizer(Tokenizer):
    """
    Deterministic token estimator using characters-per-token ratio.

    Used for dry-run because no embedding model or tiktoken is configured.
    """

    def __init__(self, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return max(1, int(len(text) / self.chars_per_token + 0.999))

    def encode_length(self, text: str) -> int:
        return self.count(text)


SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence split preserving content."""
    stripped = text.strip()
    if not stripped:
        return []
    parts = SENTENCE_PATTERN.split(stripped)
    return [part.strip() for part in parts if part.strip()]


def hard_token_split(text: str, *, max_tokens: int, tokenizer: Tokenizer) -> list[str]:
    """Last-resort split by approximate token windows without breaking mid-word when possible."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and tokenizer.count(candidate) > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


CLAUSE_BOUNDARY = re.compile(r"(?<=\.)\s+(?=\d+(?:\.\d+)+\s+)")


def _split_on_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    return parts if len(parts) > 1 else []


def _split_on_clauses(text: str) -> list[str]:
    parts = [part.strip() for part in CLAUSE_BOUNDARY.split(text) if part.strip()]
    return parts if len(parts) > 1 else []


def _group_bounded_parts(
    parts: list[str],
    *,
    max_tokens: int,
    tokenizer: Tokenizer,
) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for part in parts:
        candidate = " ".join([*current, part]).strip()
        if current and tokenizer.count(candidate) > max_tokens:
            groups.append(" ".join(current))
            current = [part]
        else:
            current.append(part)
    if current:
        groups.append(" ".join(current))
    return groups


def _shift_split_backward(
    head: str,
    tail: str,
    *,
    max_tokens: int,
    min_meaningful_tokens: int,
    tokenizer: Tokenizer,
) -> list[str] | None:
    """Move words from head into tail until tail is meaningful or head would become too small."""
    head_words = head.split()
    tail_words = tail.split()
    if not head_words or not tail_words:
        return None

    best: list[str] | None = None
    while head_words:
        moved = head_words.pop()
        tail_words.insert(0, moved)
        new_head = " ".join(head_words)
        new_tail = " ".join(tail_words)
        head_tokens = tokenizer.count(new_head)
        tail_tokens = tokenizer.count(new_tail)
        if head_tokens <= max_tokens and tail_tokens >= min_meaningful_tokens:
            best = [new_head, new_tail]
            break
        if head_tokens < min_meaningful_tokens:
            break

    if best is not None:
        return best

    combined = f"{head} {tail}".strip()
    if tokenizer.count(combined) <= max_tokens:
        return [combined]
    return None


def _fix_orphan_tail(
    parts: list[str],
    *,
    max_tokens: int,
    min_meaningful_tokens: int,
    tokenizer: Tokenizer,
) -> list[str]:
    if len(parts) < 2:
        return parts

    tail_tokens = tokenizer.count(parts[-1])
    if tail_tokens >= min_meaningful_tokens:
        return parts

    head = parts[-2]
    tail = parts[-1]
    combined = f"{head} {tail}".strip()

    for boundary_parts in (
        split_sentences(combined),
        _split_on_paragraphs(combined),
        _split_on_clauses(combined),
    ):
        if len(boundary_parts) > 1:
            regrouped = _group_bounded_parts(boundary_parts, max_tokens=max_tokens, tokenizer=tokenizer)
            if len(regrouped) >= 1 and all(tokenizer.count(part) <= max_tokens for part in regrouped):
                if len(regrouped) == 1 or tokenizer.count(regrouped[-1]) >= min_meaningful_tokens:
                    return [*parts[:-2], *regrouped]

    shifted = _shift_split_backward(
        head,
        tail,
        max_tokens=max_tokens,
        min_meaningful_tokens=min_meaningful_tokens,
        tokenizer=tokenizer,
    )
    if shifted:
        return [*parts[:-2], *shifted]

    if tokenizer.count(combined) <= max_tokens:
        return [*parts[:-2], combined]

    return parts


def smart_hard_token_split(
    text: str,
    *,
    max_tokens: int,
    min_meaningful_tokens: int,
    tokenizer: Tokenizer,
) -> list[str]:
    """
    Hard token split with orphan tail prevention.

    When the final segment is below min_meaningful_tokens, redistribute using
    sentence/paragraph/clause boundaries or shift the split point backward.
    """
    parts = hard_token_split(text, max_tokens=max_tokens, tokenizer=tokenizer)
    while len(parts) > 1:
        updated = _fix_orphan_tail(
            parts,
            max_tokens=max_tokens,
            min_meaningful_tokens=min_meaningful_tokens,
            tokenizer=tokenizer,
        )
        if updated == parts:
            break
        parts = updated
    return parts
