from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.dependencies import get_channel_intake
from app.helpers.chat_api import chat_result_payload
from app.helpers.channel import webhook_secret_matches
from app.schemas import ChannelInboundResponse, ChatResponse
from app.services.channels.intake import ChannelIntake

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
        return PlainTextResponse(hub_challenge or "")
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("/whatsapp", response_model=ChannelInboundResponse)
def whatsapp_inbound(
    payload: dict,
    intake: ChannelIntake = Depends(get_channel_intake),
    x_webhook_secret: str | None = Header(default=None),
) -> ChannelInboundResponse:
    _require_webhook_secret(x_webhook_secret)
    results = intake.handle_whatsapp(payload)
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
