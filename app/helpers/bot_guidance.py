from __future__ import annotations

import re

from app.helpers.conversation_extract import (
    looks_like_acquisition_intent,
    looks_like_device_support_issue,
    looks_like_general_hearing_concern,
    looks_like_purchase_intent,
)
from app.helpers.conversation_turn import (
    is_acknowledgement_only,
    is_casual_conversation,
    is_greeting_only,
    is_short_no,
    is_short_yes,
)
from app.helpers.query_normalize import looks_like_informational_question, looks_like_knowledge_request, strip_query_fillers
from app.services.conversation.models import ConversationGoal, ConversationState

CAPABILITY_REPLY = (
    "I can answer questions about ears, hearing, and hearing aids, help you buy, "
    "solve customer queries, and connect you with our team. What would you like help with?"
)

INSUFFICIENT_REDIRECT_REPLY = (
    "I can help with questions about ear health, hearing, hearing aids, and Earkart. "
    "Ask me about those, or I can connect you with our team."
)

UNCLEAR_REDIRECT_REPLY = (
    "I'm here for ear health, hearing, hearing aids, buying, customer queries, "
    "and connecting you with our team. Tell me what you need in a few words."
)

_BOT_CAPABILITY_RE = re.compile(
    r"(?:"
    r"\bwhat can you (?:do|help)\b"
    r"|\bhow can you help\b"
    r"|\bwhat do you do\b"
    r"|\bhow do you help\b"
    r"|\bwhat are you (?:for|here for)\b"
    r"|\byour (?:capabilities|services)\b"
    r"|\baap kya (?:kar|karo|karte|kar sakte)"
    r"|\bkya kar(?:o+|te| sakte)"
    r")",
    re.IGNORECASE,
)
_KNOWLEDGE_WRAPPER_RE = re.compile(
    r"tell me about|\bwhat is\b|\bwhat's\b|\bwarranty\b|\bprice\b|\bbattery\b",
    re.IGNORECASE,
)
_NUDGE_RE = re.compile(
    r"^(?:sir|ji|bhai|please|pls|plz|bolo+|bol+|suno+|hello\s+bolo)[.!\s]*$",
    re.IGNORECASE,
)
_WHERE_ARE_YOU_RE = re.compile(
    r"^(?:aap|tum|tu)?\s*(?:kahan|kahaan|kaha)\s+ho[.!?\s]*$"
    r"|^(?:where are you)(?:\s+(?:now|right now))?[.!?\s]*$",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"\b(?:"
    r"hearing|hear|ear|ears|aid|aids|tiny|bluup|radius|earkart|"
    r"warranty|battery|price|product|model|clinic|audiolog|"
    r"sunai|sunaai|awaz|aawaz|avaaz|kaan|machine"
    r")\b",
    re.IGNORECASE,
)
_STRETCHED_LETTER_RE = re.compile(r"([A-Za-z])\1{2,}")


def capability_reply() -> str:
    return CAPABILITY_REPLY


def unclear_redirect_reply() -> str:
    return UNCLEAR_REDIRECT_REPLY


def _normalized(message: str) -> str:
    text = strip_query_fillers(message or "")
    text = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_domain_topic(message: str) -> bool:
    return bool(_DOMAIN_RE.search(message or ""))


def looks_like_capability_question(message: str) -> bool:
    text = _normalized(message)
    if not text:
        return False
    if _KNOWLEDGE_WRAPPER_RE.search(text) and looks_like_knowledge_request(text):
        return False
    if _has_domain_topic(text):
        return False
    return bool(_BOT_CAPABILITY_RE.search(text) or _NUDGE_RE.match(text))


def looks_like_unclear_user_message(message: str) -> bool:
    raw = (message or "").strip()
    if not raw:
        return False
    if is_greeting_only(raw) or is_casual_conversation(raw):
        return False
    if is_short_yes(raw) or is_short_no(raw) or is_acknowledgement_only(raw):
        return False
    if looks_like_capability_question(raw):
        return False
    if looks_like_knowledge_request(raw) or looks_like_informational_question(raw):
        if _WHERE_ARE_YOU_RE.match(_normalized(raw)):
            return True
        return False
    if looks_like_purchase_intent(raw) or looks_like_acquisition_intent(raw):
        return False
    if looks_like_device_support_issue(raw) or looks_like_general_hearing_concern(raw):
        return False
    text = _normalized(raw)
    if _WHERE_ARE_YOU_RE.match(text) or _NUDGE_RE.match(text):
        return True
    if _has_domain_topic(text) or _has_domain_topic(raw):
        return False
    if _STRETCHED_LETTER_RE.search(raw) and len(text.split()) <= 6:
        return True
    return False


def needs_bot_guidance(message: str) -> bool:
    return looks_like_capability_question(message) or looks_like_unclear_user_message(message)


def idle_for_bot_guidance(state: ConversationState) -> bool:
    if state.awaiting_field or state.lead_collection_active or state.support_collection_active:
        return False
    if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES, ConversationGoal.SUPPORT}:
        return False
    return True
