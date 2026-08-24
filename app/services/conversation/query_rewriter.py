from __future__ import annotations

import re

from app.services.conversation.models import ConversationGoal, ConversationState
from app.helpers.query_normalize import canonicalize_knowledge_query
from app.helpers.conversation_turn import wants_more_product_info

PRONOUN_RE = re.compile(r"\b(it|its|this|that|the product)\b", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(
    r"\b(?:price|mrp|cost|warranty|battery|features?|specs?|channels?|how much)\b",
    re.IGNORECASE,
)
COMPANY_RE = re.compile(r"\bearkart\b|\bcompany\b|\bclinic\b", re.IGNORECASE)

KNOWN_PRODUCTS = (
    "Radius M16",
    "Radius 16",
    "Radius 8",
    "Radius",
    "TINY",
    "FAME SP",
    "FAME P",
    "FAME 2T",
    "FAME",
    "FORT ULTRA POWER",
    "FORT",
    "OMNI",
    "Bluup",
)


def extract_product(message: str) -> str:
    lowered = message.lower()
    for name in KNOWN_PRODUCTS:
        if name.lower() in lowered:
            return name
    return ""


def apply_named_product(state: ConversationState) -> None:
    message = state.user_message or ""
    if re.search(r"\bdifference\b|\bcompare\b|\bvs\.?\b", message, flags=re.IGNORECASE):
        return
    named = extract_product(message)
    if not named:
        return
    if state.lead_collection_active and state.awaiting_field:
        return
    explicit_switch = bool(
        re.search(
            r"\b(?:actually|instead|changed my mind|rather|i want|i'll buy|i will buy|buy|purchase|better|mean)\b",
            message,
            flags=re.IGNORECASE,
        )
    )
    if not state.product or explicit_switch:
        state.product = named
    elif state.conversation_goal == ConversationGoal.SUPPORT and named:
        state.product = named


def needs_rewrite(message: str, product: str) -> bool:
    if not product:
        return False
    if not PRONOUN_RE.search(message or ""):
        return False
    return product.lower() not in (message or "").lower()


def rewrite_query(message: str, product: str) -> str:
    text = message.strip()
    text = re.sub(
        rf"\bits\s+(.+?)(\?|$)",
        rf"the \1 of {product}\2",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bthe product\b", product, text, flags=re.IGNORECASE)
    text = re.sub(r"\bit\b", product, text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis\b", product, text, flags=re.IGNORECASE)
    text = re.sub(r"\bthat\b", product, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


class QueryRewriter:
    def apply(self, state: ConversationState) -> ConversationState:
        original = (state.user_message or "").strip()
        trace = state.trace or {}
        resolved = str(trace.get("resolved_query") or "").strip()
        if resolved and resolved.lower() != original.lower():
            state.query_rewritten = resolved
            return state
        sub_questions = trace.get("sub_questions") or []
        if sub_questions:
            first = str(sub_questions[0]).strip()
            if first and first.lower() != original.lower():
                state.query_rewritten = first
                return state
        prepared = confirmation_knowledge_query(state) or canonicalize_knowledge_query(original)
        if needs_rewrite(prepared, state.product):
            prepared = rewrite_query(prepared, state.product)
        elif should_bind_product(prepared, state.product):
            prepared = bind_product_query(prepared, state.product)
        if prepared and prepared != original:
            state.query_rewritten = prepared
        else:
            state.query_rewritten = ""
        return state


def confirmation_knowledge_query(state: ConversationState) -> str:
    product = state.product or ""
    if product and wants_more_product_info(state.user_message or ""):
        return f"What is {product}?"
    return ""


def should_bind_product(message: str, product: str) -> bool:
    if not product or not message:
        return False
    if product.lower() in message.lower():
        return False
    if COMPANY_RE.search(message):
        return False
    named = extract_product(message)
    if named and named.lower() != product.lower():
        return False
    return bool(ATTRIBUTE_RE.search(message))


def bind_product_query(message: str, product: str) -> str:
    text = message.rstrip(" ?.!")
    return f"{text} of {product}?"
