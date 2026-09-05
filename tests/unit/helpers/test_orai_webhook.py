from app.helpers.orai_webhook import parse_orai_inbound


CLOUD_TEXT = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "WABA",
            "changes": [
                {
                    "value": {
                        "messaging_product": "whatsapp",
                        "contacts": [{"profile": {"name": "Rahul"}, "wa_id": "919876543210"}],
                        "messages": [
                            {
                                "from": "919876543210",
                                "id": "wamid.1",
                                "timestamp": "1710845000",
                                "type": "text",
                                "text": {"body": "I want to buy TINY"},
                            }
                        ],
                    },
                    "field": "messages",
                }
            ],
        }
    ],
}


def test_parse_whatsapp_cloud_text() -> None:
    inbound = parse_orai_inbound(CLOUD_TEXT)
    assert len(inbound) == 1
    assert inbound[0].message == "I want to buy TINY"
    assert inbound[0].phone == "919876543210"
    assert inbound[0].user_name == "Rahul"
    assert inbound[0].channel == "whatsapp"
    assert inbound[0].origin == "whatsapp"


def test_parse_status_only_webhook_is_empty() -> None:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {"statuses": [{"id": "wamid.1", "status": "delivered"}]},
                        "field": "messages",
                    }
                ]
            }
        ],
    }
    assert parse_orai_inbound(payload) == []


def test_parse_flat_orai_payload() -> None:
    inbound = parse_orai_inbound({"from": "919876543210", "message": "Hi", "name": "Ada"})
    assert len(inbound) == 1
    assert inbound[0].message == "Hi"
    assert inbound[0].phone == "919876543210"
    assert inbound[0].user_name == "Ada"


def test_parse_non_text_is_skipped() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": "919876543210", "type": "image", "image": {"id": "x"}}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    assert parse_orai_inbound(payload) == []
