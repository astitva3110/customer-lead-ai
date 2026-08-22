from __future__ import annotations

from app.services.conversation.models import ConversationState


class InMemoryConversationRepository:
    def __init__(self) -> None:
        self._items: dict[str, ConversationState] = {}

    def get(self, conversation_id: str) -> ConversationState | None:
        state = self._items.get(conversation_id)
        if state is None:
            return None
        return ConversationState.from_dict(state.to_dict())

    def save(self, state: ConversationState) -> None:
        self._items[state.conversation_id] = ConversationState.from_dict(state.to_dict())
