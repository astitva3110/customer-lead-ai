from __future__ import annotations

import logging

from app.helpers.orai_webhook import parse_orai_inbound
from app.providers.channels.orai import send_whatsapp_reply
from app.services.conversation.models import ConversationState, InboundMessage
from app.services.conversation.orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)


class ChannelIntake:
    """Vendor webhooks enter here, then share the same conversation path as /chat."""

    def __init__(self, orchestrator: ConversationOrchestrator) -> None:
        self._orchestrator = orchestrator

    def handle_whatsapp(self, payload: dict) -> list[ConversationState]:
        return [self._run(inbound) for inbound in parse_orai_inbound(payload)]

    def handle_inbound(self, inbound: InboundMessage) -> ConversationState:
        return self._run(inbound)

    def _run(self, inbound: InboundMessage) -> ConversationState:
        result = self._orchestrator.handle(
            inbound.conversation_id,
            inbound.message,
            country=inbound.country,
            channel=inbound.channel,
            origin=inbound.origin,
            phone=inbound.phone,
            user_name=inbound.user_name,
        )
        if inbound.channel == "whatsapp":
            try:
                send_whatsapp_reply(to=inbound.phone or "", text=result.response or "")
            except Exception:
                logger.exception("whatsapp outbound send failed")
        return result
