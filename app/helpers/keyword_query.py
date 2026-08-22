from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def fts_or_query(query: str) -> str | None:
    """Build a PostgreSQL tsquery that ORs alphanumeric tokens for candidate recall."""
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(query) if token]
    if not tokens:
        return None
    unique = list(dict.fromkeys(tokens))
    return " | ".join(unique)
