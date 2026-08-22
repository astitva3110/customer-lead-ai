from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.conversation.models import TurnIntent
from app.helpers.generation_json import extract_json_object

TURN_INTENTS = {
    "KNOWLEDGE",
    "LEAD",
    "SALES",
    "SUPPORT",
    "CONTEXT_UPDATE",
    "GENERAL",
    "MIXED",
    "LEAD_INTENT",
    "SUPPORT_INTENT",
    "PROVIDE_INFORMATION",
    "CONFIRMATION",
    "ACTION",
    "UNKNOWN",
}

TURN_UNDERSTANDING_SYSTEM = (
    "Return JSON only. Classify the CURRENT user message in a hearing-aid sales/support chat. "
    "Schema: turn_intent (KNOWLEDGE|LEAD|SUPPORT|CONTEXT_UPDATE|GENERAL|MIXED|CONFIRMATION|ACTION), "
    "needs_rag (bool), lead_intent (bool), support_intent (bool), "
    "information_updates {name, phone, city, product, issue}, "
    "user_context_updates (object of short string facts the USER volunteered), "
    "explicit_action (create_lead|create_ticket|null), confidence (0-1). "
    "User facts are conversation context, never company knowledge. "
    "Informational purchase questions (why buy, benefits, price, warranty) are KNOWLEDGE not LEAD. "
    "Actionable purchase/contact intent is LEAD. Product problems are SUPPORT."
)


@dataclass
class TurnUnderstanding:
    turn_intent: str = TurnIntent.UNKNOWN
    needs_rag: bool = False
    lead_intent: bool = False
    support_intent: bool = False
    information_updates: dict[str, str | None] = field(
        default_factory=lambda: {
            "name": None,
            "phone": None,
            "city": None,
            "product": None,
            "issue": None,
        }
    )
    user_context_updates: dict[str, Any] = field(default_factory=dict)
    explicit_action: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_turn_understanding(raw: str) -> TurnUnderstanding | None:
    payload = extract_json_object(raw)
    if not payload:
        return None
    intent = str(payload.get("turn_intent") or "").upper()
    if intent not in TURN_INTENTS:
        return None
    if intent == "LEAD" or intent == "SALES":
        intent = TurnIntent.LEAD_INTENT
    elif intent == "SUPPORT":
        intent = TurnIntent.SUPPORT_INTENT
    updates = payload.get("information_updates") or {}
    if not isinstance(updates, dict):
        updates = {}
    context = payload.get("user_context_updates") or {}
    if not isinstance(context, dict):
        context = {}
    action = payload.get("explicit_action")
    if action not in {"create_lead", "create_ticket", None, ""}:
        action = None
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return TurnUnderstanding(
        turn_intent=intent,
        needs_rag=bool(payload.get("needs_rag")),
        lead_intent=bool(payload.get("lead_intent")),
        support_intent=bool(payload.get("support_intent")),
        information_updates={
            "name": _clean(updates.get("name")),
            "phone": _clean(updates.get("phone")),
            "city": _clean(updates.get("city")),
            "product": _clean(updates.get("product")),
            "issue": _clean(updates.get("issue")),
        },
        user_context_updates={str(key): str(value) for key, value in context.items() if value},
        explicit_action=action or None,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
