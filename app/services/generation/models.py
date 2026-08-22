from __future__ import annotations

from dataclasses import dataclass

INSUFFICIENT_INFORMATION_MESSAGE = (
    "I don't have enough information in the available knowledge base to answer that accurately."
)


@dataclass(frozen=True)
class GenerationResult:
    grounded: bool
    answer: str
    source_ids: list[str]


def ungrounded_fallback() -> GenerationResult:
    return GenerationResult(
        grounded=False,
        answer=INSUFFICIENT_INFORMATION_MESSAGE,
        source_ids=[],
    )
