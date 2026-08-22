from typing import Protocol


class LLMProvider(Protocol):
    """Application-facing generation contract. No vendor SDK, no model names."""

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    @property
    def is_configured(self) -> bool: ...
