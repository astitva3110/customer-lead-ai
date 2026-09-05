from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import settings
from app.dependencies import get_orchestrator
from app.main import app
from app.schemas import ChatRequest
from app.services.conversation.models import LeadStatus
from tests.unit.conversation.fakes import make_orchestrator
from tests.unit.helpers.test_orai_webhook import CLOUD_TEXT


def test_whatsapp_does_not_ask_for_known_phone() -> None:
    orchestrator, _, _, lead_tool, _, _ = make_orchestrator()
    first = orchestrator.handle(
        None,
        "connect me to sales",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
        user_name="Rahul",
    )
    assert first.phone == "+919876543210"
    assert first.user_name == "Rahul"
    assert first.conversation_id.startswith("whatsapp:")
    assert first.awaiting_field == "city"
    assert "number" not in first.response.lower()
    done = orchestrator.handle(
        first.conversation_id,
        "Delhi",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
    )
    assert done.lead_status == LeadStatus.CREATED
    assert lead_tool.leads[0].phone == "+919876543210"


def test_later_turn_keeps_whatsapp_source() -> None:
    orchestrator, *_ = make_orchestrator()
    first = orchestrator.handle(
        None,
        "What is TINY?",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
    )
    later = orchestrator.handle(first.conversation_id, "thanks")
    assert later.channel == "whatsapp"
    assert later.origin == "whatsapp"
    assert later.phone == "+919876543210"


def test_frontend_origin_stays_on_state() -> None:
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle(
        "web-1",
        "What is TINY?",
        origin="earkart.in",
    )
    assert result.channel == "web"
    assert result.origin == "earkart.in"


def test_whatsapp_webhook_runs_conversation() -> None:
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post("/channels/whatsapp", json=CLOUD_TEXT)
        assert response.status_code == 200
        payload = response.json()
        assert payload["accepted"] is True
        assert payload["results"][0]["conversation_id"].startswith("whatsapp:")
        assert payload["results"][0]["response"]
    finally:
        app.dependency_overrides.clear()


def test_chat_accepts_frontend_origin() -> None:
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "What is TINY?", "origin": "earkart.com"},
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"]
    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_unknown_origin() -> None:
    try:
        ChatRequest(message="What is TINY?", origin="evil.test")
    except ValidationError:
        return
    raise AssertionError("unknown origin must be rejected")


def test_meta_channel_is_reserved() -> None:
    client = TestClient(app)
    response = client.post("/channels/meta", json={"message": "hi"})
    assert response.status_code == 501


def test_whatsapp_webhook_verify_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "whatsapp_verify_token", "verify-me")
    client = TestClient(app)
    denied = client.get(
        "/channels/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "abc"},
    )
    assert denied.status_code == 403
    allowed = client.get(
        "/channels/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "abc"},
    )
    assert allowed.status_code == 200
    assert allowed.text == "abc"


def test_whatsapp_webhook_rejects_bad_secret(monkeypatch) -> None:
    monkeypatch.setattr(settings, "channel_webhook_secret", "expected-secret")
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        denied = client.post("/channels/whatsapp", json={"message": "Hi", "from": "919876543210"})
        assert denied.status_code == 401
        allowed = client.post(
            "/channels/whatsapp",
            json={"message": "Hi", "from": "919876543210"},
            headers={"X-Webhook-Secret": "expected-secret"},
        )
        assert allowed.status_code == 200
    finally:
        app.dependency_overrides.clear()
