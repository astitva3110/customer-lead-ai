from __future__ import annotations

from typing import Any, Protocol


class KnowledgeService(Protocol):
    def retrieve_knowledge(self, query: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]: ...
