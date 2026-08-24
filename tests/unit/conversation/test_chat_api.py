from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_orchestrator
from tests.unit.conversation.fakes import make_orchestrator


def test_chat_api_returns_mode_and_conversation_id() -> None:
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post("/chat", json={"message": "What is Radius M16?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "KNOWLEDGE"
        assert payload["conversation_id"]
        assert payload["response"]
        assert payload["answer"] == payload["response"]
        assert "sources" in payload
        assert "debug_trace_id" not in payload
        assert "langgraph" not in payload
    finally:
        app.dependency_overrides.clear()


def test_chat_page_is_public() -> None:
    client = TestClient(app)
    response = client.get("/chat")
    assert response.status_code == 200
    assert "earKART Chat" in response.text
