from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.helpers.crm_lead import crm_city_value

logger = logging.getLogger(__name__)


class EarkartCrmClient:
    """POST leads to earKART CRM (legacy PHP post_to_crm equivalent)."""

    def __init__(
        self,
        *,
        url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._url = (url or settings.crm_lead_url or "").strip()
        self._timeout = float(
            timeout_seconds if timeout_seconds is not None else settings.crm_lead_timeout_seconds
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._url)

    def create_lead(
        self,
        *,
        names: str,
        phone: str,
        source: str,
        email: str = "",
        problem: str = "",
        city: str = "null",
    ) -> None:
        if not self.is_configured:
            raise RuntimeError("CRM lead URL is not configured")
        payload = {
            "names": names,
            "phone": phone,
            "city": crm_city_value(city),
            "source": source,
            "email": email,
            "problem": problem,
        }
        response = httpx.post(
            self._url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        logger.info(
            "crm lead created source=%s phone=%s status=%s",
            source,
            phone[-4:] if len(phone) >= 4 else phone,
            response.status_code,
        )
