from __future__ import annotations

import re

_PLUS_91 = re.compile(r"^\+?91")


def parse_phone(raw: str) -> tuple[str, str] | None:
    """Return (normalized E.164-ish phone, country code) or None if invalid."""
    text = (raw or "").strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) < 10:
        return None
    if _PLUS_91.match(text.replace(" ", "")) or (digits.startswith("91") and len(digits) >= 12):
        local = digits[-10:]
        if len(local) != 10 or not local[0] in "6789":
            return None
        return f"+91{local}", "IN"
    if len(digits) == 10 and digits[0] in "6789":
        return f"+91{digits}", "IN"
    if 10 <= len(digits) <= 15:
        return f"+{digits}", "UNKNOWN"
    return None
