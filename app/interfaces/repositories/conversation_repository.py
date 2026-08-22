from __future__ import annotations

from typing import Protocol

from app.services.conversation.models import ConversationState


class ConversationRepository(Protocol):
    def get(self, conversation_id: str) -> ConversationState | None: ...

    def save(self, state: ConversationState) -> None: ...
