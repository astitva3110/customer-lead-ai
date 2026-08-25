from __future__ import annotations

import re

from app.helpers.phone import extract_and_validate_phone
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
_ISSUE_HINTS = (
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
    "issue",
    "problem",
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


def looks_like_issue(message: str) -> bool:
    lowered = (message or "").lower()
    return any(token in lowered for token in _ISSUE_HINTS)


def extract_issue(message: str) -> str:
    if looks_like_issue(message):
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
    r"please have someone call",
    r"want someone from sales",
)
TICKET_REQUEST_PATTERNS = (
    r"create a (?:support )?ticket",
    r"open a (?:support )?ticket",
    r"raise a (?:support )?ticket",
    r"please (?:create|open) a ticket",
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


def merge_user_context(existing: dict | None, updates: dict[str, str]) -> dict[str, str]:
    merged = dict(existing or {})
    merged.update({key: value for key, value in updates.items() if value})
    return merged
