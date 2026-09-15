from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from app.helpers.conversation_extract import (
    looks_like_device_support_issue,
    looks_like_general_hearing_concern,
    looks_like_product_interest_request,
    looks_like_requested_field_reply,
)
from app.helpers.conversation_history import compact_recent_history
from app.helpers.conversation_turn import accepting_knowledge_followup, last_assistant_text
from app.helpers.generation_json import extract_json_object
from app.helpers.query_normalize import looks_like_informational_question, looks_like_knowledge_request
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

SEMANTIC_ROUTER_MARKER = "You are the decision engine for a customer-support and sales chatbot."

SEMANTIC_ROUTES = {"LEAD", "KNOWLEDGE", "SUPPORT", "GENERAL"}
DECISION_INTENTS = {"knowledge", "lead", "support", "conversation", "other"}
DECISION_ACTIONS = {
    "answer_knowledge",
    "start_lead",
    "continue_lead",
    "start_support",
    "continue_support",
    "conversation",
    "reject",
}
ACTION_TO_ROUTE = {
    "answer_knowledge": "KNOWLEDGE",
    "start_lead": "LEAD",
    "continue_lead": "LEAD",
    "start_support": "SUPPORT",
    "continue_support": "SUPPORT",
    "conversation": "GENERAL",
    "reject": "GENERAL",
}
def _catalog_prompt_block() -> str:
    lines: list[str] = []
    for item in compact_product_catalog():
        aliases = ", ".join(str(alias) for alias in (item.get("aliases") or [])) or "-"
        lines.append(f"- {item['name']} | family={item['family']} | aliases={aliases}")
    return "\n".join(lines)


SEMANTIC_ROUTER_SYSTEM = f"""{SEMANTIC_ROUTER_MARKER}

Your job is NOT to answer the user.
Your job is to decide the NEXT ACTION for the application.

You receive:
- CURRENT_MESSAGE
- CONVERSATION_STATE
- RECENT_CONVERSATION

Return ONLY valid JSON:

{{
  "intent": "knowledge | lead | support | conversation | other",
  "action": "answer_knowledge | start_lead | continue_lead | start_support | continue_support | conversation | reject",
  "needs_rewrite": true,
  "product": null
}}

ROUTING PRIORITY

When multiple interpretations are possible, classify by the user's EXPLICIT ACTION/GOAL
before classifying by product mentions or topic.

Highest priority:
1. EXPLICIT LEAD REQUEST
2. ACTIVE LEAD WORKFLOW CONTINUATION
3. EXPLICIT SUPPORT REQUEST
4. ACTIVE SUPPORT WORKFLOW CONTINUATION
5. PURCHASE / BUYING INTENT
6. KNOWLEDGE QUESTION
7. GENERAL CONVERSATION

A product name does NOT determine intent by itself.

1. EXPLICIT LEAD REQUEST
Route to lead / start_lead when the user explicitly asks to create, start, submit,
register, generate, or proceed with a sales lead or enquiry.

Examples:
- "I want to create a sales lead"
- "create sale lead"
- "ok create sale lead"
- "please create a lead"
- "register me for this product"
- "submit my details to sales"
- "connect me with sales"
- "ok connect me to the sales team"
- "start the sales process"
- "proceed with the purchase"
- "yes create the lead"
- "ok create it"
- "yeah" / "ok" after the bot offered to connect the user with sales

These must NOT be routed to knowledge or conversation merely because a product name
is present in the conversation.

Never downgrade an explicit lead action into knowledge, support, or general conversation.
Connecting the user to sales is lead / start_lead, not support.

2. ACTIVE LEAD WORKFLOW CONTINUATION
If the conversation is already in lead collection and the user is providing the
requested field (name, phone, city, or other field the bot asked for), use
continue_lead + needs_rewrite=false.

Short confirmations during an active lead workflow also continue lead when they
clearly advance the lead, for example:
- "yes"
- "ok"
- "haan kar do"
- "yes, create it" (when recent conversation is clearly about creating/submitting the lead)

3. EXPLICIT SUPPORT REQUEST
Route to support / start_support when the user explicitly asks to create, open,
raise, or register a support ticket/case, or clearly asks for customer support
for a product problem.

4. ACTIVE SUPPORT WORKFLOW CONTINUATION
If the conversation is already in support collection and the user is providing
the requested field, use continue_support + needs_rewrite=false.

5. PURCHASE / BUYING INTENT
Route to lead / start_lead when the user shows buying or sales intent without an
explicit "create lead" phrase:
- wants to buy/order a product
- asks how to purchase
- asks for a demo
- expresses clear interest in buying

Examples:
- "I want to buy Radius" → lead
- "I want to create a sales lead for Radius" → lead
- "ok create sale lead" → lead

6. KNOWLEDGE QUESTION
Use knowledge / answer_knowledge when the user asks for information about:
- products, features, specifications, battery, warranty, pricing, company info, FAQs

Examples:
- "What is Radius?" → knowledge
- "How much is Radius?" → knowledge

IMPORTANT DURING LEAD OR SUPPORT COLLECTION:
If the user asks a product, feature, price, warranty, or policy question IN BETWEEN
collecting lead/support fields, route to knowledge / answer_knowledge — NOT
continue_lead or continue_support.

Example:
State: awaiting_phone for a lead
Message: "i wanna know more in details of TINY"
→ knowledge + answer_knowledge + needs_rewrite=false + product="TINY"

Example:
State: user is discussing Radius M16
Message: "What about its battery?"
→ knowledge + needs_rewrite=true + product="Radius M16"

If the user mixes buying intent with a factual question in one message, prefer
knowledge when the message is primarily asking for information.

7. GENERAL CONVERSATION
Use conversation for greetings, thanks, casual chat, or simple social messages
that are not an explicit lead/support action and not an information question.

CONTEXT
Always use conversation state and recent conversation to resolve short or ambiguous
messages.

More examples:
State: awaiting_phone for a lead
Message: "+91 9876543210"
→ continue_lead + needs_rewrite=false

State: awaiting_name for a lead
Message: "Astitva"
→ continue_lead + needs_rewrite=false

Message: "My hearing aid stopped working"
→ start_support + needs_rewrite=false

REWRITE
Set needs_rewrite=true ONLY when the message cannot be reliably searched without
conversation context.
Set needs_rewrite=false when the user's message is already a complete standalone query.

PRODUCT
Extract the product/model name only when it is explicitly stated or clearly
established by conversation context. Otherwise return null.

Known products (canonical names only; do not invent products):
{_catalog_prompt_block()}

IMPORTANT:
- Do not answer the user.
- Never answer the user.
- Never invent missing information.
- Classify explicit action/goal before product/topic.
- A product mention alone is not lead intent.
- An explicit lead/support action must not be downgraded to knowledge or conversation.
- During lead/support collection, factual/product questions in between field collection
  are knowledge, not continue_lead or continue_support.
- Current workflow state applies when the user is providing a requested field or
  clearly confirming lead/support progression.
- Return JSON only. No markdown. No explanation.
"""

SEMANTIC_ROUTER_RECOVERY = (
    "Previous output was not valid JSON matching the schema. "
    "Return ONLY one JSON object with keys intent, action, needs_rewrite, product."
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
    intent: str = ""
    action: str = ""
    needs_rewrite: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload


def build_semantic_router_prompt(state: ConversationState) -> str:
    original = (state.user_message or "").strip()
    history = compact_recent_history(state.conversation_history, limit=4, max_chars=160)
    payload = {
        "CURRENT_MESSAGE": original,
        "CONVERSATION_STATE": {
            "conversation_status": conversation_status_label(state),
            "current_status": current_status_label(state),
            "active_product": state.product or None,
            "awaiting_field": state.awaiting_field or None,
            "lead_collection_active": bool(state.lead_collection_active),
            "support_collection_active": bool(state.support_collection_active),
            "lead_status": str(state.lead_status or "") or None,
            "support_status": str(state.ticket_status or "") or None,
            "lead_stage": str(state.lead_stage or "") or None,
            "sales_interest": bool(state.sales_interest or state.lead_intent),
        },
        "RECENT_CONVERSATION": history,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_semantic_route(raw: str) -> SemanticRouteResult | None:
    payload = extract_json_object(raw)
    if not payload:
        return None
    action = str(payload.get("action") or "").strip().lower()
    intent = str(payload.get("intent") or "").strip().lower()
    decision_schema = action in DECISION_ACTIONS or intent in DECISION_INTENTS
    if decision_schema:
        if action and action not in DECISION_ACTIONS:
            return None
        if intent and intent not in DECISION_INTENTS:
            return None
        if not action:
            action = {
                "knowledge": "answer_knowledge",
                "lead": "start_lead",
                "support": "start_support",
                "conversation": "conversation",
                "other": "conversation",
            }.get(intent, "")
        if action not in DECISION_ACTIONS:
            return None
        if not intent:
            intent = {
                "answer_knowledge": "knowledge",
                "start_lead": "lead",
                "continue_lead": "lead",
                "start_support": "support",
                "continue_support": "support",
                "conversation": "conversation",
                "reject": "other",
            }.get(action, "other")
        route = ACTION_TO_ROUTE[action]
        needs_rewrite = payload.get("needs_rewrite")
        if needs_rewrite is None:
            needs_rewrite = action == "answer_knowledge"
        else:
            needs_rewrite = bool(needs_rewrite)
        product = payload.get("product")
        product_text = str(product).strip() if product not in (None, "", "null") else ""
        validated = resolve_catalog_product(product_text) if product_text else None
        product_invalid = bool(product_text) and validated is None
        if product_invalid:
            logger.info("semantic router invalid product=%s", product_text)
        try:
            confidence = float(payload.get("confidence") or 0.95)
        except (TypeError, ValueError):
            confidence = 0.95
        sales_interest = bool(payload.get("sales_interest")) or action in {"start_lead", "continue_lead"}
        return SemanticRouteResult(
            route=route,
            canonical_query=str(payload.get("canonical_query") or "").strip(),
            product=validated,
            sales_interest=sales_interest,
            diverge=bool(payload.get("diverge")),
            sub_questions=[],
            confidence=max(0.0, min(1.0, confidence)),
            product_invalid=product_invalid,
            raw=(raw or "").strip(),
            intent=intent,
            action=action,
            needs_rewrite=needs_rewrite,
        )

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
    action = {
        "KNOWLEDGE": "answer_knowledge",
        "LEAD": "start_lead",
        "SUPPORT": "start_support",
        "GENERAL": "conversation",
    }.get(route, "conversation")
    intent = {
        "KNOWLEDGE": "knowledge",
        "LEAD": "lead",
        "SUPPORT": "support",
        "GENERAL": "conversation",
    }.get(route, "other")
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
        intent=intent,
        action=action,
        needs_rewrite=None,
    )


def understanding_from_semantic(
    result: SemanticRouteResult,
    phrase: TurnUnderstanding,
    state: ConversationState,
) -> TurnUnderstanding:
    message = routing_query(state) or (state.user_message or "")
    if looks_like_product_interest_request(message):
        named = extract_product(message)
        return TurnUnderstanding(
            turn_intent=TurnIntent.LEAD_INTENT,
            needs_rag=False,
            lead_intent=True,
            support_intent=False,
            information_updates={
                "name": None,
                "phone": None,
                "city": None,
                "product": named or None,
                "issue": None,
            },
            confidence=max(result.confidence, phrase.confidence),
        )
    last = last_assistant_text(state)
    accepting_info = accepting_knowledge_followup(state.user_message or "", last)
    knowledge_turn = bool(
        phrase.needs_rag
        or phrase.turn_intent == TurnIntent.KNOWLEDGE
        or looks_like_knowledge_request(message)
        or looks_like_informational_question(message)
        or accepting_info
    )
    lead_intent = bool(result.sales_interest or result.route == "LEAD" or phrase.lead_intent)
    support_intent = bool(result.route == "SUPPORT" or (phrase.support_intent and result.route != "LEAD"))
    if accepting_info:
        return TurnUnderstanding(
            turn_intent=TurnIntent.KNOWLEDGE,
            needs_rag=True,
            lead_intent=False,
            support_intent=False,
            information_updates=dict(phrase.information_updates or {}),
            user_context_updates=phrase.user_context_updates,
            explicit_action=None,
            confidence=max(result.confidence, phrase.confidence),
        )
    if result.action in {"continue_lead", "continue_support"} and knowledge_turn:
        intent = TurnIntent.KNOWLEDGE
        needs_rag = True
        if result.action == "continue_lead":
            lead_intent = True
        elif result.action == "continue_support":
            support_intent = True
    elif result.action == "continue_lead":
        if looks_like_requested_field_reply(state, state.user_message or ""):
            intent = TurnIntent.PROVIDE_INFORMATION
            needs_rag = False
            support_intent = False
            lead_intent = True
        elif extract_product(message):
            intent = TurnIntent.KNOWLEDGE
            needs_rag = True
            lead_intent = True
        else:
            intent = TurnIntent.GENERAL
            needs_rag = False
            lead_intent = True
    elif result.action == "continue_support":
        if looks_like_requested_field_reply(state, state.user_message or ""):
            intent = TurnIntent.PROVIDE_INFORMATION
            needs_rag = False
            support_intent = True
        elif knowledge_turn or extract_product(message):
            intent = TurnIntent.KNOWLEDGE
            needs_rag = True
            support_intent = True
        else:
            intent = TurnIntent.SUPPORT_INTENT
            needs_rag = False
            support_intent = True
    elif result.action == "reject":
        intent = TurnIntent.GENERAL
        needs_rag = False
    elif result.route == "LEAD":
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
    elif result.action in {"conversation", "reject"}:
        action = None
    elif result.route == "SUPPORT" and action == "create_lead":
        if phrase.explicit_action == "create_lead" or phrase.lead_intent:
            action = "create_lead"
            intent = TurnIntent.LEAD_INTENT
            needs_rag = False
            lead_intent = True
            support_intent = False
        else:
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
    message = routing_query(state) or (state.user_message or "")
    if result is None:
        return "invalid_json"
    if result.confidence < threshold:
        return "low_confidence"
    if result.route == "SUPPORT" and (
        phrase.explicit_action == "create_lead"
        or (phrase.lead_intent and not phrase.support_intent)
        or looks_like_product_interest_request(message)
    ):
        return "prefer_lead"
    if result.route == "GENERAL" and (
        phrase.support_intent or phrase.turn_intent == TurnIntent.SUPPORT_INTENT
    ):
        return "prefer_support"
    if result.route == "GENERAL" and _phrase_is_factual(phrase, state):
        return "prefer_knowledge"
    if (
        result.route == "SUPPORT"
        and phrase.turn_intent == TurnIntent.KNOWLEDGE
        and phrase.needs_rag
        and accepting_knowledge_followup(state.user_message or "", last_assistant_text(state))
    ):
        return "prefer_knowledge_followup"
    if result.route == "SUPPORT" and looks_like_general_hearing_concern(message):
        if not looks_like_device_support_issue(message):
            return "prefer_general_hearing_knowledge"
    if (
        result.route == "SUPPORT"
        and phrase.turn_intent == TurnIntent.KNOWLEDGE
        and phrase.needs_rag
        and looks_like_general_hearing_concern(message)
        and not looks_like_device_support_issue(message)
    ):
        return "prefer_general_hearing_knowledge"
    return None


def apply_mixed_knowledge_override(
    result: SemanticRouteResult,
    phrase: TurnUnderstanding,
    state: ConversationState,
) -> SemanticRouteResult:
    """Keep buy/support intent but answer the current fact question through RAG."""
    message = routing_query(state) or (state.user_message or "")
    knowledge_turn = bool(
        phrase.needs_rag
        or phrase.turn_intent == TurnIntent.KNOWLEDGE
        or looks_like_knowledge_request(message)
        or looks_like_informational_question(message)
    )
    collecting = result.action in {"continue_lead", "continue_support", "start_lead"}
    if knowledge_turn and collecting:
        return _knowledge_with_sales(result, state, sales=result.action != "continue_support")
    if result.route != "LEAD" or not phrase.needs_rag:
        return result
    return _knowledge_with_sales(result, state, sales=True)


def _knowledge_with_sales(result: SemanticRouteResult, state: ConversationState, *, sales: bool) -> SemanticRouteResult:
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
        sales_interest=sales or result.sales_interest,
        diverge=result.diverge if len(questions) > 1 else False,
        sub_questions=questions,
        confidence=result.confidence,
        product_invalid=result.product_invalid,
        raw=result.raw,
        intent="knowledge",
        action="answer_knowledge",
        needs_rewrite=True if result.needs_rewrite is None else result.needs_rewrite,
    )


def apply_canonical_query(state: ConversationState, result: SemanticRouteResult) -> None:
    if result.needs_rewrite is False:
        return
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
