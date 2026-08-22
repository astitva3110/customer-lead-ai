from typing import Any, Protocol


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int | None = None) -> list[Any]: ...
