from __future__ import annotations

import hmac
from typing import Any

from app.helpers.conversation_extract import normalize_person_name
from app.helpers.phone import apply_phone_to_state, digits_only, extract_and_validate_phone

CHANNELS = frozenset({"web", "whatsapp", "meta"})
WEB_ORIGINS = frozenset({"dashboard", "earkart.com", "earkart.in"})
MESSAGING_ORIGINS = frozenset({"whatsapp", "meta"})
ALL_ORIGINS = WEB_ORIGINS | MESSAGING_ORIGINS
TRUSTED_PHONE_CHANNELS = MESSAGING_ORIGINS


def normalize_channel(value: str | None) -> str:
    text = (value or "").strip().lower()
    return text if text in CHANNELS else ""


def normalize_origin(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    text = text.removeprefix("https://").removeprefix("http://")
    text = text.split("/")[0]
    if text.startswith("www."):
        text = text[4:]
    if text in ALL_ORIGINS:
        return text
    return ""


def resolve_request_source(
    channel: str | None = None,
    origin: str | None = None,
    source: str | None = None,
) -> tuple[str, str]:
    resolved_origin = normalize_origin(origin) or normalize_origin(source)
    resolved_channel = normalize_channel(channel)
    if not resolved_channel and resolved_origin in MESSAGING_ORIGINS:
        resolved_channel = resolved_origin
    elif not resolved_channel and resolved_origin in WEB_ORIGINS:
        resolved_channel = "web"
    if resolved_channel in MESSAGING_ORIGINS and not resolved_origin:
        resolved_origin = resolved_channel
    return resolved_channel, resolved_origin


def phone_identity_key(raw_phone: str | None) -> str:
    text = (raw_phone or "").strip()
    if not text:
        return ""
    result = extract_and_validate_phone(text)
    if result.valid:
        return result.e164
    digits = digits_only(text)
    return f"+{digits}" if digits else ""


def resolve_conversation_id(
    conversation_id: str | None,
    channel: str,
    phone: str | None,
) -> str:
    existing = (conversation_id or "").strip()
    if existing:
        return existing
    if channel in MESSAGING_ORIGINS:
        key = phone_identity_key(phone)
        if key:
            return f"{channel}:{key}"
    return ""


def apply_known_phone(state: Any, raw_phone: str | None, *, channel: str = "") -> None:
    if getattr(state, "phone", ""):
        return
    text = (raw_phone or "").strip()
    if not text:
        return
    result = extract_and_validate_phone(text)
    if result.valid:
        apply_phone_to_state(state, result)
        return
    if channel not in TRUSTED_PHONE_CHANNELS:
        return
    digits = digits_only(text)
    if not digits:
        return
    state.phone = f"+{digits}"
    if not getattr(state, "phone_country", ""):
        state.phone_country = getattr(state, "session_country", "") or "IN"


def apply_inbound_identity(
    state: Any,
    *,
    channel: str,
    origin: str,
    phone: str | None = None,
    user_name: str | None = None,
) -> None:
    if channel:
        state.channel = channel
    elif not getattr(state, "channel", ""):
        state.channel = "web"
    if origin:
        state.origin = origin
    elif not getattr(state, "origin", "") and getattr(state, "channel", "") == "web":
        state.origin = ""
    name = normalize_person_name(user_name or "")
    if name and not getattr(state, "user_name", ""):
        state.user_name = name
    apply_known_phone(state, phone, channel=getattr(state, "channel", "") or channel)


def webhook_secret_matches(provided: str | None, expected: str | None) -> bool:
    secret = (expected or "").strip()
    if not secret:
        return True
    return hmac.compare_digest((provided or "").strip(), secret)
