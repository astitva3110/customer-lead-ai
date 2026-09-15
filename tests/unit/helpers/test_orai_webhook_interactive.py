from app.helpers.orai_webhook import parse_orai_inbound


def test_parse_interactive_button_reply() -> None:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.btn",
                                    "type": "interactive",
                                    "interactive": {
                                        "type": "button_reply",
                                        "button_reply": {"id": "lead_yes", "title": "Yes"},
                                    },
                                }
                            ]
                        },
                        "field": "messages",
                    }
                ]
            }
        ],
    }
    messages = parse_orai_inbound(payload)
    assert len(messages) == 1
    assert messages[0].message == "lead_yes"
    assert messages[0].phone == "919876543210"


def test_parse_interactive_button_reply_no() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.btn.no",
                                    "type": "interactive",
                                    "interactive": {
                                        "type": "button_reply",
                                        "button_reply": {"id": "lead_no", "title": "No"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    messages = parse_orai_inbound(payload)
    assert len(messages) == 1
    assert messages[0].message == "lead_no"
