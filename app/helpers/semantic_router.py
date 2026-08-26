from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.helpers.generation_json import extract_json_object
from app.helpers.query_normalize import looks_like_informational_question
from app.helpers.state_manager import conversation_status_label
from app.helpers.turn_understanding import TurnUnderstanding
from app.services.conversation.models import ConversationState, TurnIntent
from app.services.conversation.query_rewriter import extract_product, routing_query

SEMANTIC_ROUTER_MARKER = "You are the conversation semantic router."

SEMANTIC_ROUTES = {"LEAD", "KNOWLEDGE", "GENERAL"}

SEMANTIC_ROUTER_SYSTEM = f"""{SEMANTIC_ROUTER_MARKER}
Return ONLY valid JSON. Do not answer the user. Do not explain.

Routes:
- LEAD: user wants to buy/purchase/get a product, talk to sales, request a callback, or give sales-lead info, and the current turn is NOT a fact question.
- KNOWLEDGE: factual/informational questions (product features, warranty, battery, Bluetooth, price, company, people, policy, service). Prefer KNOWLEDGE whenever company knowledge should be retrieved. If the message shows buy intent AND asks a fact question, route=KNOWLEDGE and sales_interest=true.
- GENERAL: ONLY casual chat (hi, hello, how are you, thanks, okay, good morning). Never GENERAL for facts. If unsure whether it is factual, use KNOWLEDGE.

Rules:
- Do not treat purchase intent as a sub-question.
- diverge=true only when there are multiple independent information questions.
- Do not reset a SALE conversation_status.
- product: canonical name if known, else null.
- sub_questions: self-contained knowledge questions only; omit buy/sales actions.
- If the user asks for catalog/all-product prices (products you have, pricelist),
  do not inject active_product into sub_questions.

Schema:
{{"route":"LEAD|KNOWLEDGE|GENERAL","product":null,"sales_interest":false,"diverge":false,"sub_questions":[],"confidence":0.0}}
"""

SEMANTIC_ROUTER_RECOVERY = (
    "Previous output was not valid JSON matching the schema. "
    "Return ONLY one JSON object with keys route, product, sales_interest, diverge, sub_questions, confidence."
)


def recovery_user_prompt(original_user: str, previous: str) -> str:
    preview = (previous or "").strip()[:240]
    return f"{original_user}\n{SEMANTIC_ROUTER_RECOVERY}\nprevious_output={preview}"


@dataclass
class SemanticRouteResult:
    route: str
    product: str | None = None
    sales_interest: bool = False
    diverge: bool = False
    sub_questions: list[str] = field(default_factory=list)
    confidence: float = 0.0
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
        "active_product": state.product or None,
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
    sub_questions = payload.get("sub_questions") or []
    if not isinstance(sub_questions, list):
        sub_questions = []
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return SemanticRouteResult(
        route=route,
        product=product_text or None,
        sales_interest=bool(payload.get("sales_interest")),
        diverge=bool(payload.get("diverge")),
        sub_questions=[str(item).strip() for item in sub_questions if str(item).strip()],
        confidence=max(0.0, min(1.0, confidence)),
        raw=(raw or "").strip(),
    )


def understanding_from_semantic(
    result: SemanticRouteResult,
    phrase: TurnUnderstanding,
    state: ConversationState,
) -> TurnUnderstanding:
    lead_intent = bool(result.sales_interest or result.route == "LEAD" or phrase.lead_intent)
    if result.route == "LEAD":
        intent = TurnIntent.LEAD_INTENT
        needs_rag = False
    elif result.route == "KNOWLEDGE":
        intent = TurnIntent.KNOWLEDGE
        needs_rag = True
    else:
        intent = TurnIntent.GENERAL
        needs_rag = False
    updates = dict(phrase.information_updates or {})
    product = result.product or extract_product(routing_query(state)) or updates.get("product")
    if product:
        updates["product"] = product
    action = phrase.explicit_action if result.route == "LEAD" else None
    return TurnUnderstanding(
        turn_intent=intent,
        needs_rag=needs_rag,
        lead_intent=lead_intent,
        support_intent=False,
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
    if result.route == "GENERAL" and _phrase_is_factual(phrase, state):
        return "prefer_knowledge"
    if result.route == "LEAD" and phrase.needs_rag:
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
        rewritten = routing_query(state)
        if rewritten:
            questions = [rewritten]
    return SemanticRouteResult(
        route="KNOWLEDGE",
        product=product,
        sales_interest=True,
        diverge=result.diverge if len(questions) > 1 else False,
        sub_questions=questions,
        confidence=result.confidence,
        raw=result.raw,
    )


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
