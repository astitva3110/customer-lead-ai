from app.helpers.conversation_extract import looks_like_city_value, looks_like_requested_field_reply
from app.services.conversation.models import ConversationState


def test_city_with_pincode_is_recognized() -> None:
    for message in ("Gurugram-122006", "Gurugram 122006", "Jaipur", "Noida"):
        assert looks_like_city_value(message), message


def test_city_pincode_counts_as_requested_field_reply() -> None:
    state = ConversationState(awaiting_field="city", lead_collection_active=True)
    assert looks_like_requested_field_reply(state, "Gurugram-122006")
