from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_trace_id() -> str:
    """ULID-style id. No extra dependency."""
    timestamp_ms = int(time.time() * 1000)
    extra = int.from_bytes(os.urandom(10), "big")
    value = (timestamp_ms << 80) | (extra & ((1 << 80) - 1))
    chars: list[str] = []
    for _ in range(26):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))
