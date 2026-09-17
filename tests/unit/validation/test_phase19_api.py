from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_orchestrator
from tests.conftest import chat_service_headers
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


@contextmanager
def _client():
    orchestrator, *_ = make_graph_stack(knowledge=FakeKnowledge())
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        yield TestClient(app), orchestrator
    finally:
        app.dependency_overrides.clear()


def test_valid_message_schema() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "What is Radius M16?"}, headers=chat_service_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["conversation_id"]
        assert payload["mode"] == "KNOWLEDGE"
        assert payload["response"]
        assert payload["answer"] == payload["response"]
        assert isinstance(payload["sources"], list)


def test_missing_conversation_id_is_created() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "What is TINY?"}, headers=chat_service_headers())
        assert response.status_code == 200
        assert response.json()["conversation_id"]


def test_existing_conversation_id_persists() -> None:
    with _client() as (client, _):
        first = client.post(
            "/chat",
            json={"conversation_id": "api-1", "message": "I want to buy TINY."},
            headers=chat_service_headers(),
        )
        second = client.post(
            "/chat",
            json={"conversation_id": "api-1", "message": "+91 9876543210"},
            headers=chat_service_headers(),
        )
        assert first.json()["mode"] == "LEAD"
        assert second.json()["mode"] == "LEAD"
        assert second.json()["conversation_id"] == "api-1"


def test_empty_message_rejected() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": ""}, headers=chat_service_headers())
        assert response.status_code == 422


def test_invalid_payload_rejected() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"msg": "hello"}, headers=chat_service_headers())
        assert response.status_code == 422


def test_very_long_message_rejected() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "x" * 4001}, headers=chat_service_headers())
        assert response.status_code == 422


def test_unicode_hinglish_accepted() -> None:
    with _client() as (client, _):
        response = client.post(
            "/chat",
            json={"message": "TINY kya hai? सुनने की मशीन"},
            headers=chat_service_headers(),
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"]