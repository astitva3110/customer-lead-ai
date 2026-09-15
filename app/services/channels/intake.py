from __future__ import annotations

import logging

from app.helpers.orai_webhook import parse_orai_inbound
from app.helpers.quick_replies import quick_replies_from_trace
from app.helpers.whatsapp_webhook_dedup import claim_inbound_message
from app.providers.channels.orai import send_whatsapp_reply
from app.services.conversation.models import ConversationState, InboundMessage
from app.services.conversation.orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)


class ChannelIntake:
    """Vendor webhooks enter here, then share the same conversation path as /chat."""

    def __init__(self, orchestrator: ConversationOrchestrator) -> None:
        self._orchestrator = orchestrator

    def handle_whatsapp(self, payload: dict) -> list[ConversationState]:
        inbound_messages = parse_orai_inbound(payload)
        if not inbound_messages:
            logger.info("whatsapp webhook ignored: no inbound text messages in payload")
            return []
        results: list[ConversationState] = []
        for inbound in inbound_messages:
            message_id = (inbound.external_message_id or "").strip()
            if message_id and not claim_inbound_message(message_id):
                logger.info(
                    "whatsapp inbound skipped duplicate wamid=%s message=%s",
                    message_id,
                    inbound.message,
                )
                continue
            logger.info(
                "whatsapp inbound parsed wamid=%s phone=%s user=%s message=%s",
                message_id or "-",
                inbound.phone or "",
                inbound.user_name or "",
                inbound.message,
            )
            results.append(self._run(inbound))
        return results

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
            reply = (result.response or "").strip()
            logger.info(
                "whatsapp reply ready conversation_id=%s chars=%s preview=%s",
                result.conversation_id,
                len(reply),
                reply[:200],
            )
            try:
                send_whatsapp_reply(
                    to=inbound.phone or "",
                    text=reply,
                    quick_replies=quick_replies_from_trace(result),
                )
            except Exception:
                logger.exception("whatsapp outbound send failed")
        return result
