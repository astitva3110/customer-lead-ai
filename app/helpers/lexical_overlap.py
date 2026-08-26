from __future__ import annotations

import math
import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "could",
    "does",
    "for",
    "hello",
    "hey",
    "hi",
    "how",
    "i",
    "in",
    "is",
    "just",
    "know",
    "like",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "tell",
    "the",
    "to",
    "u",
    "us",
    "want",
    "we",
    "what",
    "would",
    "you",
}


_TERM_SYNONYMS = {
    "price": ("mrp", "cost"),
    "cost": ("mrp", "price"),
    "mrp": ("price", "cost"),
}


def tokenize_query_terms(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOPWORDS]


def _term_in_haystack(term: str, haystack: set[str]) -> bool:
    if term in haystack:
        return True
    return any(synonym in haystack for synonym in _TERM_SYNONYMS.get(term, ()))


def lexical_coverage_score(query: str, text: str) -> float:
    """Fraction of query terms that appear in the candidate text. Range 0-1."""
    query_terms = tokenize_query_terms(query)
    if not query_terms:
        return 0.0
    unique_terms = list(dict.fromkeys(query_terms))
    haystack = set(_TOKEN_PATTERN.findall(text.lower()))
    hits = sum(1 for term in unique_terms if _term_in_haystack(term, haystack))
    return hits / len(unique_terms)


def document_idf(documents: list[str]) -> dict[str, float]:
    """IDF over the current candidate pool. Rare terms outrank FAQ boilerplate."""
    n = len(documents)
    if n == 0:
        return {}
    df: dict[str, int] = {}
    for document in documents:
        for term in set(_TOKEN_PATTERN.findall(document.lower())):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def idf_weighted_coverage_score(query: str, text: str, idf: dict[str, float], *, default_idf: float) -> float:
    unique_terms = list(dict.fromkeys(tokenize_query_terms(query)))
    if not unique_terms:
        return 0.0
    haystack = set(_TOKEN_PATTERN.findall(text.lower()))
    weights = [idf.get(term, default_idf) for term in unique_terms]
    denom = sum(weights)
    if denom <= 0:
        return lexical_coverage_score(query, text)
    numer = sum(
        weight
        for term, weight in zip(unique_terms, weights)
        if _term_in_haystack(term, haystack)
    )
    return numer / denom


def default_idf_value(document_count: int) -> float:
    return math.log((max(document_count, 0) + 1) / 1) + 1.0

