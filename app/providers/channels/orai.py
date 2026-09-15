from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import settings
from app.helpers.phone import digits_only

logger = logging.getLogger(__name__)


def send_whatsapp_reply(
    *,
    to: str,
    text: str,
    quick_replies: list[dict[str, str]] | None = None,
) -> None:
    """Send a session reply through Orai. URL and API key come from `.env`."""
    to_digits = digits_only(to)
    body = (text or "").strip()
    token = (settings.orai_api_key or "").strip()
    url = (settings.orai_send_url or "").strip()
    if not to_digits or not body:
        logger.info("orai send skipped: empty to=%s body_chars=%s", bool(to_digits), len(body))
        return
    if not token or not url:
        if token or url:
            logger.warning("orai send skipped: set both ORAI_SEND_URL and ORAI_API_KEY in .env")
        return
    payload = _build_payload(to_digits=to_digits, body=body, quick_replies=quick_replies)
    headers = _send_headers(url, token)
    logger.info(
        "orai send request url=%s headers=%s payload=%s",
        url,
        _masked_headers(headers),
        payload,
    )
    response = httpx.post(url, json=payload, headers=headers, timeout=15.0)
    response_preview = (response.text or "").strip()[:500]
    if response.is_error:
        logger.warning(
            "orai send failed status=%s body=%s",
            response.status_code,
            response_preview,
        )
        return
    logger.info(
        "orai send ok to=%s status=%s body=%s",
        to_digits[-4:].rjust(4, "*"),
        response.status_code,
        response_preview or "[empty]",
    )


def _build_payload(
    *,
    to_digits: str,
    body: str,
    quick_replies: list[dict[str, str]] | None,
) -> dict[str, Any]:
    buttons = _whatsapp_buttons(quick_replies)
    if buttons:
        return {
            "messaging_product": "whatsapp",
            "to": f"+{to_digits}",
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body[:1024]},
                "action": {"buttons": buttons},
            },
        }
    return {
        "messaging_product": "whatsapp",
        "to": f"+{to_digits}",
        "type": "text",
        "text": {"body": body},
    }


def _whatsapp_buttons(quick_replies: list[dict[str, str]] | None) -> list[dict[str, Any]]:
    if not quick_replies:
        return []
    buttons: list[dict[str, Any]] = []
    for item in quick_replies[:3]:
        reply_id = str(item.get("id") or "").strip()
        title = str(item.get("label") or "Option").strip()
        if not reply_id or not title:
            continue
        buttons.append(
            {
                "type": "reply",
                "reply": {
                    "id": reply_id[:256],
                    "title": title[:20],
                },
            }
        )
    return buttons


def _send_headers(url: str, token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if "graph.facebook.com" in url:
        headers["Authorization"] = f"Bearer {token}"
        return headers
    headers["API-KEY"] = token
    return headers


def _masked_headers(headers: dict[str, str]) -> dict[str, str]:
    masked = dict(headers)
    for key in ("API-KEY", "Authorization"):
        value = (masked.get(key) or "").strip()
        if not value:
            continue
        if len(value) <= 12:
            masked[key] = "[set]"
        else:
            masked[key] = f"{value[:8]}...{value[-4:]}"
    return masked
