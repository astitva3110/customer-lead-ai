"""Pure text normalization helpers for evaluation matching."""

from __future__ import annotations

import re


def normalize_match_text(text: str) -> str:
    """Collapse whitespace and lowercase for stable substring checks."""
    return " ".join(text.lower().split())


def split_answer_sentences(answer: str, *, min_length: int = 20) -> list[str]:
    parts = re.split(r"[.!?;\n]+", answer)
    return [part.strip() for part in parts if len(part.strip()) >= min_length]


def grounding_fact_matches(fact: str, combined: str) -> bool:
    """Match a grounding fact against normalized text, with | and / alternatives."""
    normalized_fact = normalize_match_text(fact)
    if normalized_fact in combined:
        return True
    if "|" in fact:
        return any(
            normalize_match_text(part.strip()) in combined
            for part in fact.split("|")
            if part.strip()
        )
    if "/" in fact:
        return any(
            normalize_match_text(part.strip()) in combined
            for part in fact.split("/")
            if part.strip()
        )
    tokens = [token for token in re.findall(r"[a-z0-9]+", normalized_fact) if len(token) >= 5]
    if tokens and any(token in combined for token in tokens):
        return True
    return False
