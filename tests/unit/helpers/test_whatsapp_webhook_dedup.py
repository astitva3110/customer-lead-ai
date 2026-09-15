from app.helpers.orai_webhook import is_status_only_payload, parse_orai_inbound, status_event_keys
from app.helpers.whatsapp_webhook_dedup import (
    claim_inbound_message,
    claim_status_event,
    reset_whatsapp_webhook_dedup,
)
from tests.unit.helpers.test_orai_webhook import CLOUD_TEXT


def setup_function() -> None:
    reset_whatsapp_webhook_dedup()


def test_parse_cloud_message_includes_wamid() -> None:
    inbound = parse_orai_inbound(CLOUD_TEXT)
    assert inbound[0].external_message_id == "wamid.1"


def test_claim_inbound_message_deduplicates() -> None:
    assert claim_inbound_message("wamid.abc") is True
    assert claim_inbound_message("wamid.abc") is False


def test_claim_status_event_deduplicates() -> None:
    assert claim_status_event("wamid.out", "delivered") is True
    assert claim_status_event("wamid.out", "delivered") is False
    assert claim_status_event("wamid.out", "read") is True


def test_status_only_payload_detection() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "statuses": [{"id": "wamid.1", "status": "delivered", "recipient_id": "919876543210"}]
                        }
                    }
                ]
            }
        ]
    }
    assert is_status_only_payload(payload) is True
    assert status_event_keys(payload) == ["wamid.1:delivered"]


def test_duplicate_status_keys_are_detected() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "statuses": [
                                {"id": "wamid.1", "status": "delivered"},
                                {"id": "wamid.1", "status": "delivered"},
                            ]
                        }
                    }
                ]
            }
        ]
    }
    keys = status_event_keys(payload)
    assert keys == ["wamid.1:delivered", "wamid.1:delivered"]
