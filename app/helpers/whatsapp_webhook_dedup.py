from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_SEEN_INBOUND: dict[str, float] = {}
_SEEN_STATUS: dict[str, float] = {}
_TTL_SECONDS = 24 * 60 * 60
_MAX_ENTRIES = 10_000


def _prune(store: dict[str, float], now: float) -> None:
    expired = [key for key, seen_at in store.items() if now - seen_at > _TTL_SECONDS]
    for key in expired:
        store.pop(key, None)
    if len(store) <= _MAX_ENTRIES:
        return
    oldest = sorted(store.items(), key=lambda item: item[1])
    for key, _seen_at in oldest[: len(store) - _MAX_ENTRIES]:
        store.pop(key, None)


def claim_inbound_message(message_id: str) -> bool:
    """Return True when this WhatsApp message id should be processed."""
    key = (message_id or "").strip()
    if not key:
        return True
    now = time.monotonic()
    with _LOCK:
        _prune(_SEEN_INBOUND, now)
        if key in _SEEN_INBOUND:
            return False
        _SEEN_INBOUND[key] = now
        return True


def claim_status_event(status_id: str, status: str) -> bool:
    """Return True when this delivery-status webhook should be handled."""
    event_id = (status_id or "").strip()
    event_status = (status or "").strip().lower()
    if not event_id:
        return True
    key = f"{event_id}:{event_status}"
    now = time.monotonic()
    with _LOCK:
        _prune(_SEEN_STATUS, now)
        if key in _SEEN_STATUS:
            return False
        _SEEN_STATUS[key] = now
        return True


def reset_whatsapp_webhook_dedup() -> None:
    """Test helper."""
    with _LOCK:
        _SEEN_INBOUND.clear()
        _SEEN_STATUS.clear()
