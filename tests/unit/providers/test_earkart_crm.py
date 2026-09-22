from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from app.providers.crm.earkart_crm import EarkartCrmClient


def test_earkart_crm_posts_expected_payload() -> None:
    client = EarkartCrmClient(url="https://crm.example/lead2", timeout_seconds=5.0)
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()

    with patch("app.providers.crm.earkart_crm.httpx.post", return_value=response) as post:
        client.create_lead(
            names="Ada",
            phone="9876543210",
            source="web",
            problem="Radius M16",
        )

    post.assert_called_once()
    assert post.call_args.kwargs["json"] == {
        "names": "Ada",
        "phone": "9876543210",
        "city": "null",
        "source": "web",
        "email": "",
        "problem": "Radius M16",
    }
    assert post.call_args.kwargs["timeout"] == 5.0


def test_earkart_crm_passes_city_when_provided() -> None:
    client = EarkartCrmClient(url="https://crm.example/lead2")
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()

    with patch("app.providers.crm.earkart_crm.httpx.post", return_value=response) as post:
        client.create_lead(
            names="Ada",
            phone="9876543210",
            source="web",
            city="Gurugram",
        )

    assert post.call_args.kwargs["json"]["city"] == "Gurugram"


def test_earkart_crm_raises_when_url_missing() -> None:
    client = EarkartCrmClient(url="")
    try:
        client.create_lead(names="", phone="1", source="web")
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_earkart_crm_propagates_http_errors() -> None:
    client = EarkartCrmClient(url="https://crm.example/lead2")
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    with patch("app.providers.crm.earkart_crm.httpx.post", return_value=response):
        try:
            client.create_lead(names="Ada", phone="9876543210", source="whatsapp")
            raised = False
        except httpx.HTTPStatusError:
            raised = True

    assert raised
