from __future__ import annotations

import re

from app.helpers.hearing_symptom_normalize import (
    is_device_hearing_symptom,
    is_personal_hearing_difficulty,
)
from app.helpers.phone import extract_and_validate_phone, looks_like_phone_attempt
from app.helpers.query_normalize import looks_like_informational_question, looks_like_knowledge_request
from app.services.conversation.query_rewriter import extract_product

_NAME_RE = re.compile(
    r"(?:my name is|i(?:'m| am))\s+([A-Za-z]{2,30})(?=\s+from\b|\s+and\b|\s*,|\s*\.|$)",
    re.IGNORECASE,
)
_CITY_RE = re.compile(r"\b(?:from|in)\s+([A-Z][a-zA-Z]{2,40})\b")
_NAME_BLOCK = frozenset(
    {
        "interested",
        "using",
        "looking",
        "trying",
        "calling",
        "going",
        "not",
        "in",
        "from",
        "the",
        "a",
        "an",
        "here",
        "there",
    }
)
_CITY_BLOCK = frozenset(
    {
        "Earkart",
        "Buying",
        "Purchase",
        "Order",
        "This",
        "That",
        "India",
        "Hearing",
        "Radius",
        "TINY",
    }
)
_HEARING_HEALTH_RE = re.compile(
    r"(?:"
    r"\b(?:i(?:'m| am)?|i have|i've|having|have got|got)\s+(?:a\s+)?hearing\s+(?:problem|issue|trouble|loss)"
    r"|\b(?:my|our)\s+hearing\s+(?:problem|issue|trouble|loss|is\s+(?:bad|poor|worse))"
    r"|\bhearing\s+(?:problem|issue|trouble|loss)\b"
    r"|\bsunai\s+(?:problem|issue|trouble|loss|nahi|kam)\b"
    r"|\b(?:can't|cannot|can not)\s+hear\b"
    r"|\b(?:difficulty|trouble)\s+hearing\b"
    r")",
    re.IGNORECASE,
)
_DEVICE_CONTEXT_RE = re.compile(
    r"\b(?:hearing aid|hearing aids|device|charger|earbud|earbuds|tiny|bluup|radius|earkart)\b",
    re.IGNORECASE,
)
_DEVICE_ISSUE_HINTS = (
    "not working",
    "isn't working",
    "isnt working",
    "stopped working",
    "broken",
    "low sound",
    "low volume",
    "very low",
    "sound is low",
    "too quiet",
    "not powering",
    "no power",
    "won't turn on",
    "wont turn on",
    "not turning on",
    "not turning off",
    "no turning",
    "doesn't turn on",
    "doesnt turn on",
    "not turn on",
    "not turn off",
    "won't charge",
    "not charging",
    "no sound",
    "troubleshoot",
)


def extract_phone_from_text(message: str, default_region: str | None = None) -> tuple[str, str] | None:
    result = extract_and_validate_phone(message, default_region)
    if not result.valid:
        return None
    return result.e164, result.country


def normalize_person_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        return ""
    return name.title() if name.islower() else name


def extract_name(message: str) -> str:
    if re.search(r"\bwho(?:'s|\s+is)\b", message or "", flags=re.IGNORECASE):
        return ""
    match = _NAME_RE.search(message or "")
    if not match:
        return ""
    name = match.group(1).strip()
    if name.lower() in _NAME_BLOCK:
        return ""
    if extract_product(name):
        return ""
    return normalize_person_name(name)


def extract_city(message: str) -> str:
    match = _CITY_RE.search(message or "")
    if not match:
        return ""
    city = match.group(1).strip()
    if city in _CITY_BLOCK or extract_product(city):
        return ""
    return city


_TURN_ISSUE_RE = re.compile(
    r"\b(?:not|no|won'?t|doesn'?t|isn'?t)\s+t[uv]*r\w*\s+(?:on|off)\b",
    re.IGNORECASE,
)


_PURCHASE_RE = re.compile(
    r"want to (?:buy|purchase|order)|need to (?:buy|purchase|order)|"
    r"like to (?:buy|purchase|order)|interested in buying|"
    r"\bi(?:'ll| will) buy\b|\blet'?s buy\b",
    re.IGNORECASE,
)


def looks_like_purchase_intent(message: str) -> bool:
    return bool(_PURCHASE_RE.search(message or ""))


def looks_like_general_hearing_concern(message: str) -> bool:
    """General hearing-health statements — not device/customer-service faults."""
    text = (message or "").strip()
    if not text:
        return False
    if is_personal_hearing_difficulty(text):
        return True
    if not _HEARING_HEALTH_RE.search(text):
        return False
    if re.search(
        r"^\s*(?:what|who|where|when|why|how|tell me about|explain)\b",
        text,
        flags=re.IGNORECASE,
    ) and not re.search(r"\b(?:i|my|me|our)\b", text, flags=re.IGNORECASE):
        return False
    return True


def looks_like_hearing_health_concern(message: str) -> bool:
    return looks_like_general_hearing_concern(message)


_HEARING_CONSULTATION_RE = re.compile(
    r"(?:"
    r"\bwhich hearing aid\b"
    r"|\bwhat hearing aid\b"
    r"|\bwhich hearing aid should i (?:need|get|buy)\b"
    r"|\bwhich (?:one|product|model).{0,40}\b(?:buy|get|choose|pick|need)\b"
    r"|\bshould i (?:buy|get|choose|need).{0,40}\bhearing\b"
    r"|\b(?:buy|get|choose|need).{0,40}\bhearing aid\b"
    r"|\bneed (?:a |an )?hearing aid\b"
    r"|\bhelp me choose\b"
    r"|\brecommend (?:a |an )?hearing aid\b"
    r"|\bwhat should i (?:buy|get|need)\b"
    r"|\bhearing loss of \d+"
    r"|\b\d+\s*%?\s*(?:percent\s+)?hearing loss\b"
    r")",
    re.IGNORECASE,
)


def looks_like_hearing_consultation_need(message: str) -> bool:
    """Personal hearing-loss or product-selection guidance — trial/consultation intent."""
    text = (message or "").strip()
    if not text:
        return False
    if looks_like_general_hearing_concern(text):
        return True
    return bool(_HEARING_CONSULTATION_RE.search(text))


def looks_like_trial_request(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:please\s+)?create a trial(?:\s+for me)?\b|\bbook a (?:trial|consultation)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


_HEARING_AID_WANT_RE = re.compile(
    r"\bi want (?:a |an |the )?hearing aids?\b",
    re.IGNORECASE,
)
_HEARING_AID_ACQUIRE_RE = re.compile(
    r"\b(?:need|want|get|buy|purchase|order)\s+(?:a |an |the |ek )?hearing aids?\b",
    re.IGNORECASE,
)
_HINGLISH_ACQUISITION_RE = re.compile(
    r"(?:"
    r"\b(?:mujhe|mujhko|muje|humein|humko)\s+(?:ek\s+)?hearing aids?\s+(?:chaiye|chahiye|chahie|lena hai|chahie)\b"
    r"|\bhearing aids?\s+(?:chaiye|chahiye|chahie)\b"
    r"|\b(?:mujhe|mujhko|muje)\s+ek\s+hearing aids?\b"
    r")",
    re.IGNORECASE,
)


def looks_like_acquisition_intent(message: str) -> bool:
    """User wants to obtain or buy a hearing aid/product — not reporting a device fault."""
    text = (message or "").strip()
    if not text:
        return False
    if looks_like_knowledge_request(text) or looks_like_informational_question(text):
        return False
    if looks_like_device_support_issue(text):
        return False
    if re.search(r"\bi want to\b", text, flags=re.IGNORECASE):
        return False
    if _HEARING_AID_WANT_RE.search(text) or _HEARING_AID_ACQUIRE_RE.search(text):
        return True
    if _HINGLISH_ACQUISITION_RE.search(text):
        return True
    named = extract_product(text)
    if named:
        if re.match(rf"^\s*i want\s+{re.escape(named)}\s*[\s!.?]*$", text, flags=re.IGNORECASE):
            return True
        if re.search(
            rf"\b(?:mujhe|mujhko|muje)\s+(?:ek\s+)?{re.escape(named)}\s+(?:chaiye|chahiye|chahie)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return looks_like_purchase_intent(text)


def looks_like_product_interest_request(message: str) -> bool:
    """Quick-reply or natural phrasing that expresses purchase interest — not support."""
    return looks_like_acquisition_intent(message)


def looks_like_device_support_issue(message: str) -> bool:
    """Product/device faults sold by earKART — not general hearing-health queries."""
    text = (message or "").strip()
    if not text:
        return False
    if looks_like_general_hearing_concern(text) and not _DEVICE_CONTEXT_RE.search(text):
        return False
    if is_device_hearing_symptom(text):
        return True
    lowered = text.lower()
    if any(token in lowered for token in _DEVICE_ISSUE_HINTS):
        return True
    if _TURN_ISSUE_RE.search(text):
        return True
    if _DEVICE_CONTEXT_RE.search(text) and re.search(
        r"\b(?:not working|broken|repair|low sound|low volume|faulty|defective|issue|problem)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def looks_like_issue(message: str) -> bool:
    return looks_like_device_support_issue(message)


def extract_issue(message: str) -> str:
    if looks_like_device_support_issue(message):
        return (message or "").strip()
    return ""


CONTACT_PATTERNS = (
    r"\bcall me\b",
    r"\bcallback\b",
    r"\bcontact me\b",
    r"please call",
    r"someone (?:to )?call",
    r"arrange (?:a )?(?:callback|purchase|order)",
    r"get in touch",
    r"sales team",
    r"connect me",
    r"let'?s proceed",
    r"\bgo ahead\b",
    r"yes,?\s*connect",
    r"(?:ok|okay|yeah),?\s*connect",
    r"create (?:a |the )?(?:sales )?lead",
    r"(?:ok|okay|yes|yeah).{0,24}create(?:\s+it)?",
    r"please have someone call",
    r"want someone from sales",
    r"please create a trial(?:\s+for me)?",
    r"create a trial(?:\s+for me)?",
    r"book a (?:trial|consultation)",
)
TICKET_REQUEST_PATTERNS = (
    r"create a (?:support )?ticket",
    r"create (?:support )?ticket",
    r"open a (?:support )?ticket",
    r"raise a (?:support )?ticket",
    r"please (?:create|open) a ticket",
    r"(?:ok|okay|yeah|yes),?\s*create(?:\s+(?:a|the))?\s+(?:support\s+)?ticket",
    r"need a (?:support )?ticket",
    r"\bticket please\b",
)
_CUSTOMER_WORD = r"(?:cust(?:o|)?mm?er)"
_SUPPORT_TARGET = rf"(?:{_CUSTOMER_WORD} )?(?:support|service)(?: team)?"
SUPPORT_CONTACT_PATTERNS = (
    rf"talk to (?:your )?{_SUPPORT_TARGET}",
    rf"speak to (?:your )?{_SUPPORT_TARGET}",
    rf"contact (?:your )?{_SUPPORT_TARGET}",
    rf"reach (?:your )?{_SUPPORT_TARGET}",
    rf"connect me (?:to|with) (?:your )?{_SUPPORT_TARGET}",
    rf"want to (?:talk|speak|contact)(?: to)? (?:your )?{_SUPPORT_TARGET}",
    rf"need to (?:talk|speak)(?: to)? (?:your )?{_SUPPORT_TARGET}",
    rf"(?:talk|speak) (?:to )?(?:your )?{_SUPPORT_TARGET}",
    rf"make contact (?:to|with) (?:your |the )?{_SUPPORT_TARGET}",
    rf"(?:contact|contect) (?:to |with )?(?:your |the )?{_SUPPORT_TARGET}",
    rf"(?:contact|contect) me (?:to|with) (?:your |the )?{_SUPPORT_TARGET}",
    rf"connect (?:me )?(?:to|with) (?:your |the )?{_SUPPORT_TARGET}",
)
# repair, repir, reapir, and similar typos
_REPAIR_WORD = r"re(?:p(?:ai?r|a?ir)|apir)"
SUPPORT_ESCALATION_PATTERNS = (
    rf"want to {_REPAIR_WORD}",
    rf"need (?:to )?{_REPAIR_WORD}",
    r"need (?:your )?(?:help|assistance|assist)",
    r"need assi?tance",
    r"\bplease help\b",
)
_FATHER_USES_RE = re.compile(
    r"(?:my |our )?(father|mother|dad|mom|parent)\s+(?:currently )?uses?\s+([A-Za-z][A-Za-z0-9\-]{1,40})",
    re.IGNORECASE,
)
_FOR_RELATIVE_RE = re.compile(
    r"\bfor (?:my |our )?(father|mother|dad|mom|parent|him|her)\b",
    re.IGNORECASE,
)
_SMALLER_RE = re.compile(r"\b(smaller|small hearing aid|prefer small)\b", re.IGNORECASE)
_LASTS_DAY_RE = re.compile(
    r"lasts?\s+(?:me )?(?:almost )?(?:a )?(?:full )?day|usually lasts",
    re.IGNORECASE,
)
_TIME_CONTEXT_RE = re.compile(r"\b(since yesterday|started yesterday|yesterday)\b", re.IGNORECASE)
_I_USE_RE = re.compile(r"\bi use\b", re.IGNORECASE)


def looks_like_contact_request(message: str) -> bool:
    return any(re.search(pattern, message or "", flags=re.IGNORECASE) for pattern in CONTACT_PATTERNS)


def looks_like_ticket_request(message: str) -> bool:
    return any(re.search(pattern, message or "", flags=re.IGNORECASE) for pattern in TICKET_REQUEST_PATTERNS)


def looks_like_support_contact_request(message: str) -> bool:
    return any(re.search(pattern, message or "", flags=re.IGNORECASE) for pattern in SUPPORT_CONTACT_PATTERNS)


def looks_like_support_escalation_request(message: str) -> bool:
    return any(re.search(pattern, message or "", flags=re.IGNORECASE) for pattern in SUPPORT_ESCALATION_PATTERNS)


def extract_user_context(message: str) -> dict[str, str]:
    text = message or ""
    updates: dict[str, str] = {}
    relative = _FATHER_USES_RE.search(text)
    if relative:
        updates["relative"] = relative.group(1).lower()
        updates["relative_uses"] = relative.group(2).strip()
    if _FOR_RELATIVE_RE.search(text):
        updates.setdefault("buying_for", "relative")
    if _SMALLER_RE.search(text):
        updates["preference"] = "smaller"
    if _LASTS_DAY_RE.search(text):
        updates["battery_experience"] = "lasts all day"
    time_match = _TIME_CONTEXT_RE.search(text)
    if time_match:
        updates["time_context"] = time_match.group(1).lower()
    named = extract_product(text)
    if named and _I_USE_RE.search(text):
        updates["owns"] = named
    return updates


_COLLECTION_DIVERGE_RE = re.compile(
    r"\b(?:what|what's|how|why|who|where|when|warranty|battery|price|feature|features|"
    r"know(?:\s+more)?|tell(?:\s+me)?(?:\s+more)?|details?|deatils|information|specs?)\b",
    re.IGNORECASE,
)
_BARE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z.'-]{1,29}(?:\s+[A-Za-z][A-Za-z.'-]{1,29}){0,2}$")
_BARE_PLACE_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,40}$")
_CITY_PINCODE_RE = re.compile(
    r"^[A-Za-z][A-Za-z .'-]{1,40}[\s,-]+\d{5,6}$",
    re.IGNORECASE,
)


def looks_like_city_value(message: str) -> bool:
    """City answer during lead collection — plain name or name with pincode."""
    text = (message or "").strip()
    if not text:
        return False
    if extract_city(text):
        return True
    if extract_product(text):
        return False
    if _BARE_PLACE_RE.match(text):
        return True
    return bool(_CITY_PINCODE_RE.match(text))


def looks_like_requested_field_reply(state, message: str) -> bool:
    """True only when the user is answering the field currently being collected."""
    field = str(getattr(state, "awaiting_field", "") or "").strip()
    text = (message or "").strip()
    if not field or not text:
        return False
    if field != "issue" and _COLLECTION_DIVERGE_RE.search(text):
        return False
    if looks_like_knowledge_request(text) or looks_like_informational_question(text):
        return False
    if field in {"phone", "phone_country"}:
        region = getattr(state, "phone_country", None) or getattr(state, "session_country", None)
        return extract_phone_from_text(text, region) is not None or looks_like_phone_attempt(text)
    if field == "name":
        return bool(extract_name(text)) or bool(_BARE_NAME_RE.match(text) and not extract_product(text))
    if field == "city":
        return looks_like_city_value(text)
    if field == "product":
        return bool(extract_product(text))
    if field == "issue":
        return True
    return False


def merge_user_context(existing: dict | None, updates: dict[str, str]) -> dict[str, str]:
    merged = dict(existing or {})
    merged.update({key: value for key, value in updates.items() if value})
    return merged
