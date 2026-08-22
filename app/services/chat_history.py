from __future__ import annotations

from app.interfaces.repositories.chat_trace_repository import ChatTraceRepository
from app.domain.entities import ChatConversation, ChatConversationBrief


class ChatHistoryService:
    def __init__(self, traces: ChatTraceRepository) -> None:
        self._traces = traces

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> list[ChatConversationBrief]:
        return self._traces.list_conversations(limit=limit, offset=offset)

    def get_conversation(self, conversation_id: str) -> ChatConversation | None:
        return self._traces.get_conversation(conversation_id)
