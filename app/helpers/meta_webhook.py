from __future__ import annotations

from typing import Any

from app.services.conversation.models import InboundMessage


def parse_meta_inbound(payload: dict[str, Any] | None) -> list[InboundMessage]:
    """Placeholder for Facebook/Instagram Messenger. Fill this when the Meta payload is available."""
    del payload
    raise NotImplementedError("Meta channel adapter is not configured yet")
