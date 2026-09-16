from __future__ import annotations

from unittest.mock import Mock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.dependencies import get_orchestrator
from app.helpers.rate_limit import client_ip, limiter
from app.main import app
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


def _request(
    *,
    client_host: str = "203.0.113.10",
    forwarded_for: str | None = None,
) -> Request:
    http_request = Mock(spec=Request)
    http_request.headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
    http_request.client = Mock(host=client_host)
    return http_request


def test_client_ip_prefers_x_forwarded_for() -> None:
    request = _request(client_host="10.0.0.1", forwarded_for="198.51.100.7, 10.0.0.1")
    assert client_ip(request) == "198.51.100.7"


def test_client_ip_falls_back_to_direct_client() -> None:
    request = _request(client_host="203.0.113.10")
    assert client_ip(request) == "203.0.113.10"


def test_rate_limit_returns_429_when_exceeded() -> None:
    isolated_limiter = Limiter(key_func=get_remote_address, enabled=True, storage_uri="memory://")
    mini_app = FastAPI()
    mini_app.state.limiter = isolated_limiter
    mini_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @mini_app.get("/limited")
    @isolated_limiter.limit("1/minute")
    def limited_route(request: Request) -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(mini_app)
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 429


def test_chat_endpoint_rate_limited_when_enabled() -> None:
    orchestrator, *_ = make_graph_stack(knowledge=FakeKnowledge())
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    limiter.enabled = True
    try:
        client = TestClient(app)
        payload = {"message": "What is TINY?"}
        for _ in range(10):
            assert client.post("/chat", json=payload).status_code == 200
        assert client.post("/chat", json=payload).status_code == 429
    finally:
        limiter.enabled = settings.rate_limit_enabled
        app.dependency_overrides.clear()
