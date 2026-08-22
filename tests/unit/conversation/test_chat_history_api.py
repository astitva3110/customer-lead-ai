from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.domain.entities import UserRole
from app.services.chat_history import ChatHistoryService
from app.domain.entities import (
    ChatConversation,
    ChatConversationBrief,
    ChatTurn,
    ChatTurnLatency,
    RetrievalLayerHit,
)
from app.helpers.chat_history import conversation_detail_payload
from app.helpers.chat_trace_persist import group_hits_by_layer
from app.main import app
from app.dependencies import get_chat_history_service
from tests.unit.auth.helpers import as_user, make_user


class FakeTraceRepo:
    def __init__(self) -> None:
        self.briefs = [
            ChatConversationBrief(
                conversation_id="conv-1",
                last_trace_id="trace-2",
                last_user_message="What is TINY?",
                last_response="TINY is compact.",
                last_intent="KNOWLEDGE",
                last_total_ms=120.5,
                turn_count=2,
                updated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
            )
        ]
        self.conversations = {
            "conv-1": ChatConversation(
                conversation_id="conv-1",
                turns=[
                    ChatTurn(
                        trace_id="trace-2",
                        conversation_id="conv-1",
                        user_message="What is TINY?",
                        rewritten_query="what is tiny",
                        intent="KNOWLEDGE",
                        response="TINY is compact.",
                        backend="llamaindex-hybrid",
                        reranker_name="PassthroughReranker",
                        latency=ChatTurnLatency(
                            retrieval_ms=40.0,
                            vector_ms=11.0,
                            keyword_ms=9.0,
                            merge_ms=1.0,
                            rerank_ms=12.5,
                            total_ms=120.5,
                        ),
                        hits=[
                            RetrievalLayerHit(
                                layer="vector",
                                rank=1,
                                chunk_id="c1",
                                vector_score=0.91,
                                text_preview="TINY is compact.",
                            ),
                            RetrievalLayerHit(
                                layer="reranked",
                                rank=1,
                                chunk_id="c1",
                                rerank_score=0.88,
                                original_retrieval_rank=1,
                            ),
                        ],
                    )
                ],
            )
        }

    def save(self, trace) -> None:
        del trace

    def list_conversations(self, *, limit: int, offset: int) -> list[ChatConversationBrief]:
        del limit, offset
        return list(self.briefs)

    def get_conversation(self, conversation_id: str) -> ChatConversation | None:
        return self.conversations.get(conversation_id)


def test_group_hits_keeps_layer_order() -> None:
    hits = [
        RetrievalLayerHit(layer="reranked", rank=1, chunk_id="c1"),
        RetrievalLayerHit(layer="vector", rank=1, chunk_id="c1"),
    ]
    grouped = group_hits_by_layer(hits)
    assert list(grouped)[:5] == ["vector", "keyword", "merged", "reranked", "final"]
    assert grouped["vector"][0].chunk_id == "c1"
    assert grouped["keyword"] == []


def test_detail_payload_includes_latency_and_layers() -> None:
    repo = FakeTraceRepo()
    payload = conversation_detail_payload(repo.conversations["conv-1"])
    turn = payload["turns"][0]
    assert turn["latency"]["vector_ms"] == 11.0
    assert turn["latency"]["rerank_ms"] == 12.5
    assert turn["layers"]["vector"][0]["chunk_id"] == "c1"
    assert turn["layers"]["reranked"][0]["original_retrieval_rank"] == 1


def test_list_chats_returns_brief_items() -> None:
    admin, _password = make_user(role=UserRole.ADMIN)
    service = ChatHistoryService(FakeTraceRepo())
    app.dependency_overrides[get_chat_history_service] = lambda: service
    try:
        with as_user(admin):
            client = TestClient(app)
            response = client.get("/chats")
        assert response.status_code == 200
        payload = response.json()
        assert payload["items"][0]["conversation_id"] == "conv-1"
        assert payload["items"][0]["last_trace_id"] == "trace-2"
        assert payload["items"][0]["turn_count"] == 2
        assert payload["items"][0]["last_total_ms"] == 120.5
    finally:
        app.dependency_overrides.clear()


def test_get_chat_returns_deep_latency_and_layers() -> None:
    admin, _password = make_user(role=UserRole.ADMIN)
    service = ChatHistoryService(FakeTraceRepo())
    app.dependency_overrides[get_chat_history_service] = lambda: service
    try:
        with as_user(admin):
            client = TestClient(app)
            response = client.get("/chats/conv-1")
        assert response.status_code == 200
        payload = response.json()
        turn = payload["turns"][0]
        assert turn["trace_id"] == "trace-2"
        assert turn["latency"]["vector_ms"] == 11.0
        assert turn["latency"]["keyword_ms"] == 9.0
        assert turn["latency"]["rerank_ms"] == 12.5
        assert turn["layers"]["vector"][0]["text_preview"] == "TINY is compact."
        assert "final" in turn["layers"]
    finally:
        app.dependency_overrides.clear()


def test_get_chat_missing_returns_404() -> None:
    admin, _password = make_user(role=UserRole.ADMIN)
    service = ChatHistoryService(FakeTraceRepo())
    app.dependency_overrides[get_chat_history_service] = lambda: service
    try:
        with as_user(admin):
            client = TestClient(app)
            response = client.get("/chats/missing")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()
