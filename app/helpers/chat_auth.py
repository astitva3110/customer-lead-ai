from __future__ import annotations

import hmac


def service_token_matches(provided: str | None, expected: str | None) -> bool:
    secret = (expected or "").strip()
    if not secret:
        return False
    return hmac.compare_digest((provided or "").strip(), secret)
