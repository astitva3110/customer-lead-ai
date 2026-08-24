from __future__ import annotations

from app.helpers.conversation_history import compact_recent_history


def test_compact_recent_history_truncates_and_limits() -> None:
    history = [
        {"role": "user", "content": "x" * 300},
        {"role": "assistant", "content": "short"},
    ]
    compact = compact_recent_history(history, limit=1, max_chars=250)
    assert len(compact) == 1
    assert compact[0]["role"] == "assistant"
    assert len(compact[0]["content"]) == 5

    full = compact_recent_history(history, limit=2, max_chars=250)
    assert len(full[0]["content"]) == 250
