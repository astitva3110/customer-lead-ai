from __future__ import annotations

import json
from typing import Any

from app.helpers.channel import resolve_request_source
from app.services.conversation.models import InboundMessage

_WEBHOOK_PREVIEW_CHARS = 2000


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
    body = _cloud_message_body(item)
    if not body:
        return None
    phone = str(item.get("from") or "").strip()
    message_id = str(item.get("id") or "").strip()
    channel, origin = resolve_request_source("whatsapp", "whatsapp")
    return InboundMessage(
        message=body,
        conversation_id=None,
        channel=channel,
        origin=origin,
        phone=phone or None,
        user_name=names.get(phone) or None,
        external_user_id=phone or None,
        external_message_id=message_id or None,
    )


def _cloud_message_body(item: dict[str, Any]) -> str:
    message_type = str(item.get("type") or "text").strip().lower()
    if message_type == "text":
        text_block = item.get("text") if isinstance(item.get("text"), dict) else {}
        return str(text_block.get("body") or "").strip()
    if message_type == "interactive":
        interactive = item.get("interactive") if isinstance(item.get("interactive"), dict) else {}
        interactive_type = str(interactive.get("type") or "").strip().lower()
        if interactive_type == "button_reply":
            button = interactive.get("button_reply") if isinstance(interactive.get("button_reply"), dict) else {}
            reply_id = str(button.get("id") or "").strip()
            if reply_id:
                return reply_id
            return str(button.get("title") or "").strip()
        if interactive_type == "list_reply":
            selected = interactive.get("list_reply") if isinstance(interactive.get("list_reply"), dict) else {}
            reply_id = str(selected.get("id") or "").strip()
            if reply_id:
                return reply_id
            return str(selected.get("title") or "").strip()
    return ""


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


def summarize_orai_payload(payload: dict[str, Any] | None) -> str:
    """Short label for logs: message webhook vs delivery status vs flat test payload."""
    if not isinstance(payload, dict) or not payload:
        return "empty"
    if not payload.get("entry"):
        keys = ",".join(sorted(str(key) for key in payload))
        return f"flat keys={keys or 'none'}"

    message_count = 0
    status_count = 0
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue
            message_count += len(value.get("messages") or [])
            status_count += len(value.get("statuses") or [])
    return f"cloud messages={message_count} statuses={status_count}"


def is_status_only_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict) or not payload.get("entry"):
        return False
    message_count = 0
    status_count = 0
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue
            message_count += len(value.get("messages") or [])
            status_count += len(value.get("statuses") or [])
    return message_count == 0 and status_count > 0


def status_event_keys(payload: dict[str, Any] | None) -> list[str]:
    keys: list[str] = []
    if not isinstance(payload, dict):
        return keys
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue
            for item in value.get("statuses") or []:
                if not isinstance(item, dict):
                    continue
                status_id = str(item.get("id") or "").strip()
                status = str(item.get("status") or "").strip().lower()
                if status_id and status:
                    keys.append(f"{status_id}:{status}")
    return keys


def summarize_status_events(payload: dict[str, Any] | None) -> str:
    events: list[str] = []
    if not isinstance(payload, dict):
        return ""
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue
            for item in value.get("statuses") or []:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "").strip().lower()
                recipient = str(item.get("recipient_id") or "").strip()
                status_id = str(item.get("id") or "").strip()
                if status:
                    events.append(f"{status} to={recipient or '?'} id={status_id[-12:] or '?'}")
    return "; ".join(events)


def preview_orai_payload(payload: dict[str, Any] | None, *, limit: int = _WEBHOOK_PREVIEW_CHARS) -> str:
    """JSON preview for console logs; truncates very large webhook bodies."""
    if payload is None:
        return "null"
    if not isinstance(payload, dict):
        return repr(payload)
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        text = str(payload)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated, total_chars={len(text)})"
