from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_orchestrator
from app.helpers.chat_auth import service_token_matches
from app.domain.entities import UserRole
from app.main import app
from tests.conftest import chat_service_headers
from tests.unit.auth.helpers import auth_header, make_user, user_repo
from tests.unit.conversation.fakes import make_orchestrator


def test_service_token_matches_expected_value() -> None:
    assert service_token_matches("secret-token", "secret-token")


def test_service_token_rejects_wrong_value() -> None:
    assert not service_token_matches("wrong", "secret-token")


def test_chat_requires_authentication() -> None:
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post("/chat", json={"message": "What is TINY?"})
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_invalid_service_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "chat_service_token", "expected-token")
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "What is TINY?"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_chat_accepts_service_token() -> None:
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "What is TINY?"},
            headers=chat_service_headers(),
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"]
    finally:
        app.dependency_overrides.clear()


def test_chat_accepts_staff_jwt() -> None:
    user, _password = make_user(role=UserRole.USER)
    orchestrator, *_ = make_orchestrator()
    with user_repo([user]):
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator
        try:
            client = TestClient(app)
            response = client.post(
                "/chat",
                json={"message": "What is TINY?", "origin": "dashboard", "channel": "web"},
                headers=auth_header(user),
            )
            assert response.status_code == 200
            assert response.json()["conversation_id"]
        finally:
            app.dependency_overrides.clear()
