from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_orchestrator
from tests.unit.conversation.fakes import FakeKnowledge
from tests.unit.auth.helpers import as_user, make_user
from tests.validation.harness import make_graph_stack


@contextmanager
def _client():
    user, _password = make_user()
    orchestrator, *_ = make_graph_stack(knowledge=FakeKnowledge())
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        with as_user(user):
            yield TestClient(app), orchestrator
    finally:
        app.dependency_overrides.clear()


def test_valid_message_schema() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "What is Radius M16?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["conversation_id"]
        assert payload["mode"] == "KNOWLEDGE"
        assert payload["response"]
        assert payload["answer"] == payload["response"]
        assert isinstance(payload["sources"], list)


def test_missing_conversation_id_is_created() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "What is TINY?"})
        assert response.status_code == 200
        assert response.json()["conversation_id"]


def test_existing_conversation_id_persists() -> None:
    with _client() as (client, _):
        first = client.post("/chat", json={"conversation_id": "api-1", "message": "I want to buy TINY."})
        second = client.post("/chat", json={"conversation_id": "api-1", "message": "+91 9876543210"})
        assert first.json()["mode"] == "LEAD"
        assert second.json()["mode"] == "LEAD"
        assert second.json()["conversation_id"] == "api-1"


def test_empty_message_rejected() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422


def test_invalid_payload_rejected() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"msg": "hello"})
        assert response.status_code == 422


def test_very_long_message_rejected() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "x" * 4001})
        assert response.status_code == 422


def test_unicode_hinglish_accepted() -> None:
    with _client() as (client, _):
        response = client.post("/chat", json={"message": "TINY kya hai? सुनने की मशीन"})
        assert response.status_code == 200
        assert response.json()["conversation_id"]