import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.dependencies import get_channel_intake
from app.helpers.chat_api import chat_result_payload
from app.helpers.channel import webhook_secret_matches
from app.helpers.orai_webhook import (
    is_status_only_payload,
    preview_orai_payload,
    status_event_keys,
    summarize_orai_payload,
    summarize_status_events,
)
from app.helpers.whatsapp_webhook_dedup import claim_status_event
from app.schemas import ChannelInboundResponse, ChatResponse
from app.services.channels.intake import ChannelIntake

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/channels", tags=["channels"])


def _require_webhook_secret(x_webhook_secret: str | None) -> None:
    expected = (settings.channel_webhook_secret or "").strip()
    if not webhook_secret_matches(x_webhook_secret, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


@router.get("/whatsapp")
def verify_whatsapp_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    expected = (settings.whatsapp_verify_token or settings.channel_webhook_secret or "").strip()
    if hub_mode == "subscribe" and webhook_secret_matches(hub_verify_token, expected):
        logger.info(
            "whatsapp webhook verified mode=%s challenge_len=%s",
            hub_mode,
            len(hub_challenge or ""),
        )
        return PlainTextResponse(hub_challenge or "")
    logger.warning(
        "whatsapp webhook verify failed mode=%s token_present=%s",
        hub_mode,
        bool((hub_verify_token or "").strip()),
    )
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("/whatsapp", response_model=ChannelInboundResponse)
def whatsapp_inbound(
    payload: dict,
    intake: ChannelIntake = Depends(get_channel_intake),
    x_webhook_secret: str | None = Header(default=None),
) -> ChannelInboundResponse:
    _require_webhook_secret(x_webhook_secret)
    if is_status_only_payload(payload):
        keys = status_event_keys(payload)
        new_keys: list[str] = []
        for key in keys:
            status_id, _, status = key.partition(":")
            if claim_status_event(status_id, status):
                new_keys.append(key)
        if not new_keys:
            logger.info("whatsapp status webhook duplicate skipped keys=%s", keys)
            return ChannelInboundResponse(accepted=True, results=[])
        logger.info(
            "whatsapp status webhook received events=%s keys=%s",
            summarize_status_events(payload),
            new_keys,
        )
        return ChannelInboundResponse(accepted=True, results=[])
    logger.info(
        "whatsapp webhook received summary=%s payload=%s",
        summarize_orai_payload(payload),
        preview_orai_payload(payload),
    )
    results = intake.handle_whatsapp(payload)
    logger.info("whatsapp webhook handled inbound_messages=%s", len(results))
    return ChannelInboundResponse(
        accepted=True,
        results=[ChatResponse(**chat_result_payload(item)) for item in results],
    )


@router.post("/meta", response_model=ChannelInboundResponse)
def meta_inbound(
    payload: dict,
    x_webhook_secret: str | None = Header(default=None),
) -> ChannelInboundResponse:
    del payload
    _require_webhook_secret(x_webhook_secret)
    raise HTTPException(status_code=501, detail="Meta channel adapter is not configured yet")
