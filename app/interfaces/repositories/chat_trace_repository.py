from __future__ import annotations

from typing import Protocol

from app.services.diagnostics.models import ChatTrace
from app.domain.entities import ChatConversation, ChatConversationBrief


class ChatTraceRepository(Protocol):
    def save(self, trace: ChatTrace) -> None: ...

    def list_conversations(self, *, limit: int, offset: int) -> list[ChatConversationBrief]: ...

    def get_conversation(self, conversation_id: str) -> ChatConversation | None: ...
