from __future__ import annotations

from typing import Any

from app.helpers.channel import resolve_request_source
from app.services.conversation.models import InboundMessage


def parse_orai_inbound(payload: dict[str, Any] | None) -> list[InboundMessage]:
    """Map Orai/WhatsApp Cloud webhooks (and a flat test payload) to inbound turns."""
    if not isinstance(payload, dict) or not payload:
        return []
    if payload.get("entry"):
        return _parse_cloud_payload(payload)
    inbound = _parse_flat_payload(payload)
    return [inbound] if inbound else []


def _parse_cloud_payload(payload: dict[str, Any]) -> list[InboundMessage]:
    messages: list[InboundMessage] = []
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue
            names = _contact_names(value)
            for item in value.get("messages") or []:
                inbound = _from_cloud_message(item, names)
                if inbound:
                    messages.append(inbound)
    return messages


def _contact_names(value: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for contact in value.get("contacts") or []:
        if not isinstance(contact, dict):
            continue
        wa_id = str(contact.get("wa_id") or "").strip()
        profile = contact.get("profile") if isinstance(contact.get("profile"), dict) else {}
        name = str(profile.get("name") or "").strip()
        if wa_id and name:
            names[wa_id] = name
    return names


def _from_cloud_message(item: Any, names: dict[str, str]) -> InboundMessage | None:
    if not isinstance(item, dict):
        return None
    if str(item.get("type") or "text") != "text":
        return None
    text_block = item.get("text") if isinstance(item.get("text"), dict) else {}
    body = str(text_block.get("body") or "").strip()
    if not body:
        return None
    phone = str(item.get("from") or "").strip()
    channel, origin = resolve_request_source("whatsapp", "whatsapp")
    return InboundMessage(
        message=body,
        conversation_id=None,
        channel=channel,
        origin=origin,
        phone=phone or None,
        user_name=names.get(phone) or None,
        external_user_id=phone or None,
    )


def _parse_flat_payload(payload: dict[str, Any]) -> InboundMessage | None:
    message = str(payload.get("message") or payload.get("text") or payload.get("body") or "").strip()
    if isinstance(payload.get("text"), dict):
        message = str(payload["text"].get("body") or message).strip()
    if not message:
        return None
    phone = str(payload.get("phone") or payload.get("from") or payload.get("wa_id") or "").strip()
    name = str(payload.get("name") or payload.get("user_name") or "").strip()
    channel, origin = resolve_request_source("whatsapp", "whatsapp")
    return InboundMessage(
        message=message,
        conversation_id=str(payload.get("conversation_id") or "").strip() or None,
        channel=channel,
        origin=origin,
        phone=phone or None,
        user_name=name or None,
        country=str(payload.get("country") or "").strip() or None,
        external_user_id=phone or None,
    )
