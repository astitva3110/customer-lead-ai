from __future__ import annotations

import re

INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "jailbreak",
    "developer mode",
    "reveal your prompt",
)
ABUSE_MARKERS = ("kill yourself",)


def guardrail_check(message: str) -> str | None:
    """Return a rejection reason, or None if the message may proceed. No LLM."""
    text = (message or "").strip()
    if not text:
        return "empty"
    lowered = text.lower()
    if any(marker in lowered for marker in INJECTION_MARKERS):
        return "prompt_injection"
    if any(marker in lowered for marker in ABUSE_MARKERS):
        return "unsupported"
    if len(text) > 4000:
        return "too_long"
    return None


GUARDRAIL_REJECTION_MESSAGE = "naa munna naaa"
