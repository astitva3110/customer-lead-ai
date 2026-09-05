from app.helpers.channel import (
    apply_inbound_identity,
    apply_known_phone,
    normalize_origin,
    phone_identity_key,
    resolve_conversation_id,
    resolve_request_source,
    webhook_secret_matches,
)
from app.services.conversation.models import ConversationState


def test_resolve_frontend_origins() -> None:
    assert resolve_request_source(origin="dashboard") == ("web", "dashboard")
    assert resolve_request_source(source="https://www.earkart.com/shop") == ("web", "earkart.com")
    assert resolve_request_source(source="earkart.in") == ("web", "earkart.in")
    assert resolve_request_source() == ("", "")


def test_resolve_messaging_sources() -> None:
    assert resolve_request_source(source="whatsapp") == ("whatsapp", "whatsapp")
    assert resolve_request_source(channel="meta") == ("meta", "meta")


def test_normalize_origin_rejects_unknown() -> None:
    assert normalize_origin("evil.example") == ""


def test_whatsapp_conversation_id_is_stable_from_phone() -> None:
    first = resolve_conversation_id(None, "whatsapp", "9876543210")
    second = resolve_conversation_id(None, "whatsapp", "+91 98765 43210")
    assert first == "whatsapp:+919876543210"
    assert first == second


def test_explicit_conversation_id_wins() -> None:
    assert resolve_conversation_id("keep-me", "whatsapp", "9876543210") == "keep-me"


def test_apply_known_phone_skips_when_already_set() -> None:
    state = ConversationState(phone="+919111111111")
    apply_known_phone(state, "9876543210", channel="whatsapp")
    assert state.phone == "+919111111111"


def test_trusted_channel_keeps_unvalidated_number() -> None:
    state = ConversationState()
    apply_known_phone(state, "14155552671", channel="whatsapp")
    assert state.phone == "+14155552671"


def test_web_does_not_store_invalid_phone() -> None:
    state = ConversationState()
    apply_known_phone(state, "123", channel="web")
    assert state.phone == ""


def test_apply_inbound_identity_sets_profile_name_once() -> None:
    state = ConversationState()
    apply_inbound_identity(state, channel="whatsapp", origin="whatsapp", user_name="rahul")
    apply_inbound_identity(state, channel="whatsapp", origin="whatsapp", user_name="Other")
    assert state.user_name == "Rahul"


def test_webhook_secret_open_when_unset() -> None:
    assert webhook_secret_matches(None, "") is True
    assert webhook_secret_matches("nope", "secret") is False
    assert webhook_secret_matches("secret", "secret") is True


def test_phone_identity_key() -> None:
    assert phone_identity_key("9876543210") == "+919876543210"
