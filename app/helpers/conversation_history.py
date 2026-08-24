from __future__ import annotations

DEFAULT_HISTORY_LIMIT = 6
DEFAULT_MESSAGE_CHARS = 250


def compact_recent_history(
    history: list[dict[str, str]] | None,
    *,
    limit: int = DEFAULT_HISTORY_LIMIT,
    max_chars: int = DEFAULT_MESSAGE_CHARS,
) -> list[dict[str, str]]:
    items = list(history or [])[-limit:]
    compact: list[dict[str, str]] = []
    for item in items:
        content = (item.get("content") or "").strip()
        if max_chars and len(content) > max_chars:
            content = content[:max_chars]
        compact.append({"role": str(item.get("role") or ""), "content": content})
    return compact
