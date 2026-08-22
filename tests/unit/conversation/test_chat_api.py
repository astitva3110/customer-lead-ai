from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_orchestrator
from tests.unit.conversation.fakes import make_orchestrator
from tests.unit.auth.helpers import as_user, make_user


def test_chat_api_returns_mode_and_conversation_id() -> None:
    user, _password = make_user()
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        with as_user(user):
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
