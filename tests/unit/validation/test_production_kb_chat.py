from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.dependencies import get_orchestrator
from app.main import app
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


@contextmanager
def _client(*, knowledge: FakeKnowledge | None = None):
    orchestrator, *_ = make_graph_stack(knowledge=knowledge or FakeKnowledge())
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        yield TestClient(app), orchestrator
    finally:
        app.dependency_overrides.clear()


def test_general_greeting_does_not_require_kb() -> None:
    with _client(knowledge=FakeKnowledge(chunks=[])) as (client, _):
        response = client.post("/chat", json={"message": "Hi"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["response"]
        assert payload["conversation_id"]


def test_knowledge_question_with_empty_kb_returns_gracefully() -> None:
    with _client(knowledge=FakeKnowledge(chunks=[])) as (client, _):
        response = client.post("/chat", json={"message": "What is TINY?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["response"] == INSUFFICIENT_INFORMATION_MESSAGE
        assert payload["sources"] == []
