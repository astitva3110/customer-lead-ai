from __future__ import annotations

import re

from app.services.conversation.models import ConversationState

GREETING_ONLY_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|good morning|good afternoon|good evening)"
    r"(?:[\s,!.]+(?:there|hi|hello))?"
    r"[\s!.]*$",
    re.IGNORECASE,
)
GREETING_PREFIX_RE = re.compile(
    r"^(hi|hello|hey|hiya|howdy|good morning|good afternoon|good evening)[,!.\s]+",
    re.IGNORECASE,
)
GREETING_SMALLTALK_RE = re.compile(
    r"^(?:how are you(?: doing)?|how's it going|how are u|what's up|whats up|"
    r"how do you do)[\s!.?]*$",
    re.IGNORECASE,
)
SHORT_YES_RE = re.compile(
    r"^(yes|yeah|yep|yup|sure|ok|okay|please|y|haan|acha)(?:\s+(?:plz|please|pls))?[\s!.]*$",
    re.IGNORECASE,
)
ACK_ONLY_RE = re.compile(
    r"^(ok|okay|hmm|hm|acha|right|fine|thanks|thank you|got it)[.!\s]*$",
    re.IGNORECASE,
)
PLEASANTRY_ONLY_RE = re.compile(
    r"^(?:"
    r"thanks(?:\s+(?:a\s+lot|so\s+much|very\s+much))?|"
    r"thank\s+you(?:\s+(?:so\s+much|very\s+much|a\s+lot))?|"
    r"thx|ty|cheers|"
    r"much\s+appreciated|appreciate\s+it|"
    r"you(?:'re| are)\s+welcome|"
    r"no\s+problem|np|"
    r"bye|goodbye|good\s+bye|see\s+you(?:\s+later)?|take\s+care|"
    r"have\s+a\s+(?:nice|good|great)\s+day|"
    r"nice\s+(?:to\s+)?(?:meet|talking\s+to)\s+you|"
    r"sounds\s+good|gotcha|cool|great|awesome|perfect|lovely"
    r")[.!\s]*$",
    re.IGNORECASE,
)
MORE_INFO_RE = re.compile(
    r"^(?:yes please|yeah please|yes plz|yeah plz|yes pls|yeah pls|tell me more|more please|go ahead|"
    r"sure tell me|yes tell me)[.!\s]*$",
    re.IGNORECASE,
)
SHORT_NO_RE = re.compile(r"^(no|nope|nah|not now|later|no thanks)[.!\s]*$", re.IGNORECASE)
OFFERED_INFO_RE = re.compile(
    r"know anything|would you like to know|tell you (?:more )?about|features|"
    r"what would you like to know|including features|price, and warranty|keep asking questions",
    re.IGNORECASE,
)
OFFERED_CALLBACK_RE = re.compile(
    r"arrange a callback|call you|contact you|callback\?|someone (?:to )?call|"
    r"sales team|connect you",
    re.IGNORECASE,
)
OFFERED_SUPPORT_RE = re.compile(
    r"support team|customer service|connect (?:you )?(?:with|to) (?:our |the )?support",
    re.IGNORECASE,
)
PHONE_REFUSAL_RE = re.compile(
    r"(?:don't|do not|won't|wont|not going to|prefer not to).{0,40}(?:phone|number|share)|"
    r"\bnot (?:share|give|provide).{0,20}(?:phone|number)",
    re.IGNORECASE,
)


def last_assistant_text(state: ConversationState) -> str:
    for item in reversed(state.conversation_history or []):
        if item.get("role") == "assistant":
            return item.get("content") or ""
    return ""


def recent_assistant_texts(state: ConversationState, *, limit: int = 2) -> list[str]:
    texts: list[str] = []
    for item in reversed(state.conversation_history or []):
        if item.get("role") == "assistant":
            texts.append(item.get("content") or "")
            if len(texts) >= limit:
                break
    return texts


def is_greeting_only(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if GREETING_ONLY_RE.match(text) or GREETING_SMALLTALK_RE.match(text):
        return True
    remainder = GREETING_PREFIX_RE.sub("", text, count=1).strip()
    return bool(remainder and remainder != text and GREETING_SMALLTALK_RE.match(remainder))


def strip_greeting_prefix(message: str) -> str:
    remainder = GREETING_PREFIX_RE.sub("", message or "", count=1).strip()
    return remainder or (message or "")


def has_greeting_prefix(message: str) -> bool:
    text = (message or "").strip()
    if not text or is_greeting_only(text):
        return False
    return bool(GREETING_PREFIX_RE.match(text))


def is_short_yes(message: str) -> bool:
    return bool(SHORT_YES_RE.match((message or "").strip()))


def is_acknowledgement_only(message: str) -> bool:
    return bool(ACK_ONLY_RE.match((message or "").strip()))


def is_casual_conversation(message: str) -> bool:
    """Greetings, thanks, acknowledgements, small talk, and pleasantries only."""
    text = (message or "").strip()
    if not text:
        return False
    return is_greeting_only(text) or is_acknowledgement_only(text) or bool(PLEASANTRY_ONLY_RE.match(text))


def wants_more_product_info(message: str) -> bool:
    text = (message or "").strip()
    return is_short_yes(text) or is_tell_more(text)


def is_tell_more(message: str) -> bool:
    return bool(MORE_INFO_RE.match((message or "").strip()))


def is_short_no(message: str) -> bool:
    return bool(SHORT_NO_RE.match((message or "").strip()))


def offered_product_information(assistant_text: str) -> bool:
    return bool(OFFERED_INFO_RE.search(assistant_text or ""))


def offered_callback(assistant_text: str) -> bool:
    text = assistant_text or ""
    if re.search(r"whenever you(?:'re| are) ready", text, flags=re.IGNORECASE):
        return False
    return bool(OFFERED_CALLBACK_RE.search(text))


def offered_support_help(assistant_text: str) -> bool:
    return bool(OFFERED_SUPPORT_RE.search(assistant_text or ""))


def assistant_asked_question(assistant_text: str) -> bool:
    return "?" in (assistant_text or "")


def is_phone_refusal(message: str) -> bool:
    return bool(PHONE_REFUSAL_RE.search(message or ""))
