from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from app.helpers.generation_json import extract_json_object
from app.helpers.query_normalize import looks_like_informational_question
from app.helpers.state_manager import conversation_status_label, current_status_label
from app.helpers.turn_understanding import TurnUnderstanding
from app.services.conversation.models import ConversationState, TurnIntent
from app.services.conversation.query_rewriter import (
    compact_product_catalog,
    extract_product,
    resolve_catalog_product,
    routing_query,
    usable_canonical_query,
)

logger = logging.getLogger(__name__)

SEMANTIC_ROUTER_MARKER = "You are the conversation semantic router."

SEMANTIC_ROUTES = {"LEAD", "KNOWLEDGE", "SUPPORT", "GENERAL"}


def _catalog_prompt_block() -> str:
    lines: list[str] = []
    for item in compact_product_catalog():
        aliases = ", ".join(str(alias) for alias in (item.get("aliases") or [])) or "-"
        lines.append(f"- {item['name']} | family={item['family']} | aliases={aliases}")
    return "\n".join(lines)


SEMANTIC_ROUTER_SYSTEM = f"""{SEMANTIC_ROUTER_MARKER}
Return ONLY valid JSON. Do not answer the user. Do not explain. Do not invent facts, prices, or offers.

Known products (canonical names only; do not invent products):
{_catalog_prompt_block()}

Routes:
- LEAD: purchase/sales intent for this turn (buy, purchase, get the product, talk to sales, callback) and the current turn is NOT a fact question.
- KNOWLEDGE: factual/informational questions (product features, warranty, battery, Bluetooth, specs, company, people, policy, service, catalog). Prefer KNOWLEDGE whenever company knowledge should be retrieved. If the message shows buy intent AND asks a fact question, route=KNOWLEDGE and sales_interest=true.
- SUPPORT: existing-device problems, repair, complaint, ticket, customer support for a device that is not working.
- GENERAL: ONLY casual chat (hi, hello, how are you, thanks, okay, good morning). Never GENERAL for facts. If unsure whether it is factual, use KNOWLEDGE.

Rules:
- canonical_query: clear self-contained rewrite of the user's meaning. Preserve intent. Do not add unimplied facts. Do not attach active_product to catalog-wide questions (what products do you have, pricelist).
- product: canonical catalog name if known, else null. Unknown mentions stay null; do not substitute another product.
- Do not treat purchase intent as a knowledge sub-question.
- Support follow-ups stay SUPPORT: device symptoms, misspellings (not turining on, no turning off), and short yes after troubleshooting. Never GENERAL for those.
- Short yes is not a lead-name collection unless the assistant offered a callback or support team.
- A question like "who is X" is KNOWLEDGE about a person. Do not treat X as the user's name.
- diverge=true only when there are multiple independent information questions.
- Do not reset a SALE conversation_status. Knowledge during SALE still has sales_interest=true.
- sub_questions: self-contained knowledge questions only; omit buy/sales actions.
- conversation_status is the persistent journey; current_status is the previous turn's work. Route this turn on the current message.

Schema:
{{"canonical_query":"string","route":"LEAD|KNOWLEDGE|SUPPORT|GENERAL","product":null,"sales_interest":false,"diverge":false,"sub_questions":[],"confidence":0.0}}
"""

SEMANTIC_ROUTER_RECOVERY = (
    "Previous output was not valid JSON matching the schema. "
    "Return ONLY one JSON object with keys canonical_query, route, product, sales_interest, diverge, sub_questions, confidence."
)


def recovery_user_prompt(original_user: str, previous: str) -> str:
    preview = (previous or "").strip()[:240]
    return f"{original_user}\n{SEMANTIC_ROUTER_RECOVERY}\nprevious_output={preview}"


@dataclass
class SemanticRouteResult:
    route: str
    canonical_query: str = ""
    product: str | None = None
    sales_interest: bool = False
    diverge: bool = False
    sub_questions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    product_invalid: bool = False
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload


def build_semantic_router_prompt(state: ConversationState) -> str:
    original = (state.user_message or "").strip()
    normalized = routing_query(state) or original
    last_user, last_assistant = _recent_turns(state, original)
    payload = {
        "original_message": original,
        "normalized_query": normalized,
        "conversation_status": conversation_status_label(state),
        "current_status": current_status_label(state),
        "active_product": state.product or None,
        "lead_status": str(state.lead_status or "") or None,
        "support_status": str(state.ticket_status or "") or None,
        "lead_stage": str(state.lead_stage or "") or None,
        "sales_interest": bool(state.sales_interest or state.lead_intent),
        "last_user": last_user or None,
        "last_assistant": last_assistant or None,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_semantic_route(raw: str) -> SemanticRouteResult | None:
    payload = extract_json_object(raw)
    if not payload:
        return None
    route = str(payload.get("route") or "").strip().upper()
    if route not in SEMANTIC_ROUTES:
        return None
    product = payload.get("product")
    product_text = str(product).strip() if product not in (None, "", "null") else ""
    validated = resolve_catalog_product(product_text) if product_text else None
    product_invalid = bool(product_text) and validated is None
    if product_invalid:
        logger.info("semantic router invalid product=%s", product_text)
    sub_questions = payload.get("sub_questions") or []
    if not isinstance(sub_questions, list):
        sub_questions = []
    canonical = str(payload.get("canonical_query") or "").strip()
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return SemanticRouteResult(
        route=route,
        canonical_query=canonical,
        product=validated,
        sales_interest=bool(payload.get("sales_interest")),
        diverge=bool(payload.get("diverge")),
        sub_questions=[str(item).strip() for item in sub_questions if str(item).strip()],
        confidence=max(0.0, min(1.0, confidence)),
        product_invalid=product_invalid,
        raw=(raw or "").strip(),
    )


def understanding_from_semantic(
    result: SemanticRouteResult,
    phrase: TurnUnderstanding,
    state: ConversationState,
) -> TurnUnderstanding:
    lead_intent = bool(result.sales_interest or result.route == "LEAD" or phrase.lead_intent)
    support_intent = bool(result.route == "SUPPORT" or (phrase.support_intent and result.route != "LEAD"))
    if result.route == "LEAD":
        intent = TurnIntent.LEAD_INTENT
        needs_rag = False
        support_intent = False
    elif result.route == "KNOWLEDGE":
        intent = TurnIntent.KNOWLEDGE
        needs_rag = True
    elif result.route == "SUPPORT":
        intent = TurnIntent.SUPPORT_INTENT
        needs_rag = False
        lead_intent = bool(result.sales_interest)
    else:
        intent = TurnIntent.GENERAL
        needs_rag = False
        if phrase.support_intent or phrase.turn_intent == TurnIntent.SUPPORT_INTENT:
            intent = TurnIntent.SUPPORT_INTENT
            support_intent = True
    updates = dict(phrase.information_updates or {})
    product = result.product or extract_product(routing_query(state)) or updates.get("product")
    if product:
        updates["product"] = product
    action = phrase.explicit_action
    if result.route == "KNOWLEDGE":
        action = None
    elif result.route == "GENERAL":
        action = None
    elif result.route == "SUPPORT" and action == "create_lead":
        action = phrase.explicit_action if phrase.explicit_action == "create_ticket" else None
    return TurnUnderstanding(
        turn_intent=intent,
        needs_rag=needs_rag,
        lead_intent=lead_intent,
        support_intent=support_intent,
        information_updates=updates,
        user_context_updates=phrase.user_context_updates,
        explicit_action=action,
        confidence=result.confidence,
    )


def semantic_fallback_reason(
    result: SemanticRouteResult | None,
    phrase: TurnUnderstanding,
    state: ConversationState,
    *,
    threshold: float,
) -> str | None:
    if result is None:
        return "invalid_json"
    if result.confidence < threshold:
        return "low_confidence"
    if result.route == "GENERAL" and (
        phrase.support_intent or phrase.turn_intent == TurnIntent.SUPPORT_INTENT
    ):
        return "prefer_support"
    if result.route == "GENERAL" and _phrase_is_factual(phrase, state):
        return "prefer_knowledge"
    return None


def apply_mixed_knowledge_override(
    result: SemanticRouteResult,
    phrase: TurnUnderstanding,
    state: ConversationState,
) -> SemanticRouteResult:
    """Keep buy intent but answer the current fact question through RAG."""
    if result.route != "LEAD" or not phrase.needs_rag:
        return result
    product = result.product or extract_product(routing_query(state)) or state.product or None
    questions = list(result.sub_questions)
    if not questions:
        rewritten = result.canonical_query or routing_query(state)
        if rewritten:
            questions = [rewritten]
    canonical = result.canonical_query or (questions[0] if questions else "")
    return SemanticRouteResult(
        route="KNOWLEDGE",
        canonical_query=canonical,
        product=product,
        sales_interest=True,
        diverge=result.diverge if len(questions) > 1 else False,
        sub_questions=questions,
        confidence=result.confidence,
        product_invalid=result.product_invalid,
        raw=result.raw,
    )


def apply_canonical_query(state: ConversationState, result: SemanticRouteResult) -> None:
    canonical = (result.canonical_query or "").strip()
    if not usable_canonical_query(canonical):
        return
    state.query_rewritten = canonical
    state.trace = dict(state.trace or {})
    state.trace["canonical_query"] = canonical


def _phrase_is_factual(phrase: TurnUnderstanding, state: ConversationState) -> bool:
    if phrase.turn_intent == TurnIntent.KNOWLEDGE or phrase.needs_rag:
        return True
    message = routing_query(state) or (state.user_message or "")
    return looks_like_informational_question(message)


def _recent_turns(state: ConversationState, current: str) -> tuple[str, str]:
    last_user = ""
    last_assistant = ""
    current_norm = current.strip().lower()
    for item in reversed(list(state.conversation_history or [])):
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and not last_assistant:
            last_assistant = content[:160]
        elif role == "user" and not last_user:
            if content.strip().lower() == current_norm:
                continue
            last_user = content[:160]
        if last_user and last_assistant:
            break
    return last_user, last_assistant
