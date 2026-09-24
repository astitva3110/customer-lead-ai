from __future__ import annotations

import re

from app.services.conversation.models import ChatMode, ConversationGoal, ConversationState, TurnIntent
from app.helpers.bot_guidance import (
    capability_reply,
    looks_like_capability_question,
    looks_like_unclear_user_message,
    unclear_redirect_reply,
)
from app.helpers.conversation_turn import is_greeting_only, is_short_no, is_short_yes, last_assistant_text, recent_assistant_texts
from app.helpers.user_language import format_name_suffix, localized_text

_QUESTIONNAIRE_CLOSERS = (
    "what would you like to know",
    "would you like to know anything",
    "what would you like to try next",
    "tell you more about",
    "connect you with our team",
)


def lead_discuss_reply(state: ConversationState) -> str:
    product = state.product or "that"
    if state.current_turn_intent in {TurnIntent.LEAD_INTENT, TurnIntent.SALES}:
        return (
            f"Absolutely! {product} is a great choice. 😊 "
            "I can help with any questions, and I can connect you with our team whenever you're ready."
        )
    if state.current_turn_intent == TurnIntent.CONTEXT_UPDATE and state.user_name:
        if state.city:
            return f"Nice to meet you, {state.user_name}. I've noted {state.city}."
        return f"Nice to meet you, {state.user_name}!"
    return f"Got it — I can keep helping with {product}."


def lead_ack_reply(state: ConversationState) -> str:
    name = state.user_name
    if name:
        return f"Nice to meet you, {name}!"
    return "I've noted that."


def support_ack_reply(state: ConversationState) -> str:
    if state.support_issue:
        product = state.product or "your hearing aid"
        return f"I'm sorry you're dealing with that on {product}. I'll help you work through it."
    return "I'm sorry you're dealing with that. Tell me a little about what's happening, and I'll see how I can help."


def support_troubleshooting_reply(state: ConversationState) -> str:
    if state.conversation_goal == ConversationGoal.SUPPORT and not state.support_collection_active:
        return support_ticket_offer_reply(state)
    product = state.product or "your hearing aid"
    message = (state.user_message or "").lower()
    if "repair" in message:
        return (
            f"I can help get {product} repaired. "
            "Would you like me to connect you with our team?"
        )
    if any(token in message for token in ("help", "assist", "assitance", "assistance")):
        return (
            f"I'm here to help with {product}. "
            "Would you like me to connect you with our team?"
        )
    return support_ack_reply(state)


def confirmation_info_reply(state: ConversationState) -> str:
    product = state.product or "it"
    return f"Sure — I can cover {product} whenever you're ready."


def greeting_reply(message: str = "") -> str:
    text = (message or "").strip().lower()
    offer = (
        "I can help with hearing, hearing aids, buying, and connecting you with our team. "
        "What do you need?"
    )
    if re.search(r"\b(?:namaste|namaskar|pranam)\b", text):
        return f"Namaste! {offer}"
    if re.search(r"\bhola\b", text):
        return f"Hola! {offer}"
    if re.search(r"\b(?:salaam|assalam|salam)\b", text):
        return f"Salaam! {offer}"
    if re.search(r"\bgood morning\b", text):
        return f"Good morning! {offer}"
    if re.search(r"\bgood afternoon\b", text):
        return f"Good afternoon! {offer}"
    if re.search(r"\bgood evening\b", text):
        return f"Good evening! {offer}"
    if re.search(r"\bgood night\b", text):
        return f"Good night! {offer}"
    return f"Hey! {offer}"


def price_query_reply(product: str = "") -> str:
    label = (product or "").strip()
    if label:
        topic = f"{label} pricing"
    else:
        topic = "pricing"
    return (
        f"For {topic}, please visit earkart.com or earkart.in. "
        "If you'd like to know about current offers, I can connect you with our team — just say yes."
    )


def support_ticket_offer_reply(state: ConversationState) -> str:
    product = state.product or "your hearing aid"
    return (
        f"I'm sorry you're having trouble with {product}. "
        "Would you like me to connect you with our team?"
    )


def lead_created_reply(name: str = "", *, language: str = "en") -> str:
    return localized_text(
        "lead_created",
        language,
        name=format_name_suffix(name),
    )


def ticket_created_reply(name: str = "") -> str:
    if name:
        return (
            f"Thanks, {name}. I've created the support ticket and shared the details "
            "with our team. They'll get back to you shortly."
        )
    return (
        "I've created the support ticket and shared the details with our team. "
        "They'll get back to you shortly."
    )


def conversational_fallback(state: ConversationState) -> str:
    message = state.user_message or ""
    if is_greeting_only(message):
        return _distinct(state, greeting_reply(message))
    if looks_like_capability_question(message):
        return _distinct(state, capability_reply())
    if looks_like_unclear_user_message(message):
        return _distinct(state, unclear_redirect_reply())
    in_support = state.conversation_goal == ConversationGoal.SUPPORT or state.mode == ChatMode.SUPPORT
    if is_short_yes(message):
        if in_support:
            return _distinct(state, support_troubleshooting_reply(state))
        return _distinct(state, confirmation_info_reply(state))
    if is_short_no(message):
        return _distinct(state, "No problem. We can keep going whenever you're ready.")
    intent = state.current_turn_intent
    if intent == TurnIntent.CONFIRMATION:
        if in_support:
            return _distinct(state, support_troubleshooting_reply(state))
        return _distinct(state, confirmation_info_reply(state))
    if in_support or intent == TurnIntent.SUPPORT_INTENT:
        return _distinct(state, support_troubleshooting_reply(state))
    if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES} or intent in {
        TurnIntent.LEAD_INTENT,
        TurnIntent.SALES,
    }:
        return _distinct(state, lead_discuss_reply(state))
    return _distinct(state, capability_reply())


def repeats_previous(previous: str, candidate: str) -> bool:
    last = (previous or "").strip().lower()
    text = (candidate or "").strip().lower()
    if not last or not text:
        return False
    if last == text:
        return True
    return any(token in last and token in text for token in _QUESTIONNAIRE_CLOSERS)


def repeats_recent_assistant(state: ConversationState, candidate: str, *, limit: int = 2) -> bool:
    for previous in recent_assistant_texts(state, limit=limit):
        if repeats_previous(previous, candidate):
            return True
    return False


def _distinct(state: ConversationState, candidate: str) -> str:
    last = last_assistant_text(state)
    if not repeats_previous(last, candidate):
        return candidate
    product = state.product or "it"
    return f"Sure — I can keep helping with {product}."
