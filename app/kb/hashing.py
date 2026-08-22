import hashlib
import json
from typing import Any


def sha256_hex(data: str | bytes) -> str:
    """Return lowercase hex SHA-256 digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_hash(content: str) -> str:
    """SHA-256 hash prefixed for content identity."""
    return f"sha256:{sha256_hex(content)}"


def hash_raw_payload(payload: dict[str, Any]) -> str:
    """
    Hash the raw document body deterministically.

    Uses the ``content`` field when present; otherwise serializes the full payload.
    """
    body = payload.get("content")
    if body is not None:
        return content_hash(str(body))
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return content_hash(canonical)


def hash_filename(content_hash_value: str) -> str:
    """Build immutable raw filename from a content hash."""
    digest = content_hash_value.removeprefix("sha256:")
    return f"sha256_{digest}.json"
