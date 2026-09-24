from __future__ import annotations

from dataclasses import dataclass

from app.helpers.bot_guidance import INSUFFICIENT_REDIRECT_REPLY

INSUFFICIENT_INFORMATION_MESSAGE = INSUFFICIENT_REDIRECT_REPLY


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
