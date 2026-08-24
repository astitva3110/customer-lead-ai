from __future__ import annotations

from unittest.mock import Mock

from fastapi.exceptions import RequestValidationError

from app.helpers.route_errors import (
    detail_to_message,
    http_error_log_message,
    request_label,
    validation_errors_to_message,
)


def _request(method: str = "GET", path: str = "/leads/missing", query: str = "") -> Mock:
    request = Mock()
    request.method = method
    request.url.path = path
    request.url.query = query
    request.client = Mock(host="127.0.0.1")
    return request


def test_request_label_includes_method_path_client_and_query() -> None:
    assert request_label(_request(query="status=open")) == "GET /leads/missing?status=open client=127.0.0.1"


def test_detail_to_message_formats_validation_items() -> None:
    detail = [{"loc": ("body", "email"), "msg": "field required"}]
    assert detail_to_message(detail) == "body.email: field required"


def test_http_error_log_message_is_readable() -> None:
    message = http_error_log_message(_request(), 404, "Lead not found")
    assert message == "HTTP 404 | GET /leads/missing client=127.0.0.1 detail=Lead not found"


def test_validation_errors_to_message() -> None:
    exc = RequestValidationError(
        [{"type": "missing", "loc": ("body", "message"), "msg": "Field required", "input": {}}]
    )
    assert validation_errors_to_message(exc) == "body.message: Field required"
