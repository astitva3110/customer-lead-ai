from __future__ import annotations

from app.services.conversation.models import ConversationGoal, ConversationState, TurnIntent
from app.helpers.conversation_turn import is_greeting_only, is_short_no, is_short_yes, last_assistant_text

_QUESTIONNAIRE_CLOSERS = (
    "what would you like to know",
    "would you like to know anything",
    "what would you like to try next",
    "tell you more about",
    "connect you with our sales",
)


def lead_discuss_reply(state: ConversationState) -> str:
    product = state.product or "that"
    if state.current_turn_intent in {TurnIntent.LEAD_INTENT, TurnIntent.SALES}:
        return (
            f"Absolutely! {product} is a great choice. 😊 "
            "I can help with any questions, and I can connect you with our sales team whenever you're ready."
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


def confirmation_info_reply(state: ConversationState) -> str:
    product = state.product or "it"
    return f"Sure — I can cover {product} whenever you're ready."


def greeting_reply() -> str:
    return "Hey! 👋 How can I help you today?"


def lead_created_reply(name: str = "") -> str:
    if name:
        return (
            f"Thanks, {name}! Your details have been shared with our sales team. "
            "They'll get in touch with you shortly."
        )
    return (
        "Thanks! Your details have been shared with our sales team. "
        "They'll get in touch with you shortly."
    )


def ticket_created_reply(name: str = "") -> str:
    if name:
        return (
            f"Thanks, {name}. I've created the support ticket and shared the details "
            "with our support team. They'll get back to you shortly."
        )
    return (
        "I've created the support ticket and shared the details with our support team. "
        "They'll get back to you shortly."
    )


def conversational_fallback(state: ConversationState) -> str:
    message = state.user_message or ""
    if is_greeting_only(message):
        return _distinct(state, greeting_reply())
    if is_short_yes(message):
        return _distinct(state, confirmation_info_reply(state))
    if is_short_no(message):
        return _distinct(state, "No problem. We can keep going whenever you're ready.")
    intent = state.current_turn_intent
    if intent == TurnIntent.CONFIRMATION:
        return _distinct(state, confirmation_info_reply(state))
    if state.conversation_goal == ConversationGoal.SUPPORT or intent == TurnIntent.SUPPORT_INTENT:
        return _distinct(state, support_ack_reply(state))
    if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES} or intent in {
        TurnIntent.LEAD_INTENT,
        TurnIntent.SALES,
    }:
        return _distinct(state, lead_discuss_reply(state))
    return _distinct(state, "Happy to help.")


def repeats_previous(previous: str, candidate: str) -> bool:
    last = (previous or "").strip().lower()
    text = (candidate or "").strip().lower()
    if not last or not text:
        return False
    if last == text:
        return True
    return any(token in last and token in text for token in _QUESTIONNAIRE_CLOSERS)


def _distinct(state: ConversationState, candidate: str) -> str:
    last = last_assistant_text(state)
    if not repeats_previous(last, candidate):
        return candidate
    product = state.product or "it"
    return f"Sure — I can keep helping with {product}."
