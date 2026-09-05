from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.helpers.phone import digits_only

logger = logging.getLogger(__name__)


def send_whatsapp_reply(*, to: str, text: str) -> None:
    """Send a session reply through Orai. URL and API key come from `.env`."""
    to_digits = digits_only(to)
    body = (text or "").strip()
    token = (settings.orai_api_key or "").strip()
    url = (settings.orai_send_url or "").strip()
    if not to_digits or not body:
        return
    if not token or not url:
        if token or url:
            logger.warning("orai send skipped: set both ORAI_SEND_URL and ORAI_API_KEY in .env")
        return
    payload = {
        "messaging_product": "whatsapp",
        "to": f"+{to_digits}",
        "type": "text",
        "text": {"body": body},
    }
    headers = _send_headers(url, token)
    response = httpx.post(url, json=payload, headers=headers, timeout=15.0)
    if response.is_error:
        logger.warning("orai send failed status=%s body=%s", response.status_code, response.text[:300])
        return
    logger.info("orai send ok to=%s status=%s", to_digits[-4:].rjust(4, "*"), response.status_code)


def _send_headers(url: str, token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if "graph.facebook.com" in url:
        headers["Authorization"] = f"Bearer {token}"
        return headers
    headers["API-KEY"] = token
    return headers
