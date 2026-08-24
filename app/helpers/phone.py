from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

REASON_INVALID_LENGTH = "invalid_length"
REASON_INVALID_PREFIX = "invalid_prefix"
REASON_NOT_FOUND = "not_found"
REASON_INVALID = "invalid_number"
REASON_COUNTRY_REQUIRED = "country_required"

CITY_COUNTRY_INDIA = "IN"
_VALID_MOBILE_PREFIXES = frozenset("6789")
_PHONE_SPAN = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")
_EXACT_CODES = {
    "IN": "IN",
    "IND": "IN",
    "US": "US",
    "USA": "US",
    "GB": "GB",
    "UK": "GB",
    "AU": "AU",
}
_NAME_ALIASES = {
    "INDIA": "IN",
    "INDIAN": "IN",
    "UNITED STATES": "US",
    "USA": "US",
    "AMERICA": "US",
    "UNITED KINGDOM": "GB",
    "BRITAIN": "GB",
    "ENGLAND": "GB",
    "AUSTRALIA": "AU",
}


@dataclass(frozen=True)
class PhoneValidationResult:
    valid: bool
    reason: str = ""
    normalized_value: str = ""
    raw_input: str = ""
    normalized_input: str = ""
    country: str = ""
    country_code: str = ""
    national_number: str = ""
    e164: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "valid": self.valid,
            "country": self.country,
            "reason": self.reason,
            "normalized_value": self.normalized_value,
            "normalized_input": self.normalized_input,
            "e164": self.e164,
        }
        if self.valid:
            payload["country_code"] = self.country_code
            payload["national_number"] = self.national_number
        return payload


def digits_only(raw: str) -> str:
    return re.sub(r"\D", "", raw or "")


def normalize_region(value: str | None) -> str | None:
    text = (value or "").strip().upper()
    if not text:
        return None
    compact = re.sub(r"[^A-Z]", "", text)
    if compact in _EXACT_CODES and len(compact) <= 3:
        return _EXACT_CODES[compact]
    cleaned = " ".join(re.sub(r"[^A-Z ]", "", text).split())
    if cleaned in _NAME_ALIASES:
        return _NAME_ALIASES[cleaned]
    return _EXACT_CODES.get(compact)


def default_phone_region(state: Any) -> str:
    context = ""
    user_context = getattr(state, "user_context", None) or {}
    if isinstance(user_context, dict):
        context = str(user_context.get("country") or "")
    for value in (getattr(state, "session_country", ""), context):
        region = normalize_region(value)
        if region:
            return region
    return CITY_COUNTRY_INDIA


def parse_country_reply(message: str) -> str | None:
    text = (message or "").strip()
    if not text:
        return None
    direct = normalize_region(text)
    if direct:
        return direct
    upper = text.upper()
    for alias, region in sorted(_NAME_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", upper):
            return region
    return None


def normalize_indian_mobile_digits(raw: str) -> str:
    digits = digits_only(raw)
    if len(digits) >= 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def validate_phone_number(raw: str, default_region: str | None = None) -> PhoneValidationResult:
    del default_region
    text = (raw or "").strip()
    raw_digits = digits_only(text)
    normalized = normalize_indian_mobile_digits(text)
    if not raw_digits:
        return PhoneValidationResult(valid=False, reason=REASON_NOT_FOUND, raw_input=text)
    if len(normalized) != 10:
        return PhoneValidationResult(
            valid=False,
            reason=REASON_INVALID_LENGTH,
            raw_input=text,
            normalized_input=normalized,
            country="IN",
        )
    if normalized[0] not in _VALID_MOBILE_PREFIXES:
        return PhoneValidationResult(
            valid=False,
            reason=REASON_INVALID_PREFIX,
            raw_input=text,
            normalized_input=normalized,
            country="IN",
        )
    return PhoneValidationResult(
        valid=True,
        normalized_value=normalized,
        raw_input=text,
        normalized_input=normalized,
        country="IN",
        country_code="+91",
        national_number=normalized,
        e164=f"+91{normalized}",
    )


def looks_like_phone_attempt(message: str) -> bool:
    text = (message or "").strip()
    digits = digits_only(text)
    if len(digits) >= 8:
        return True
    if len(digits) < 5:
        return False
    compact = re.sub(r"[\s\-\(\)\.+]", "", text)
    if compact.isdigit() or compact.startswith("91"):
        return True
    return bool(re.search(r"\b(number|phone|mobile|whatsapp)\b", text, flags=re.IGNORECASE))


def extract_and_validate_phone(
    message: str,
    default_region: str | None = None,
) -> PhoneValidationResult:
    result = validate_phone_number(message, default_region)
    if result.valid:
        return result
    for match in _PHONE_SPAN.finditer(message or ""):
        span_result = validate_phone_number(match.group(0), default_region)
        if span_result.valid:
            return span_result
    if looks_like_phone_attempt(message):
        return result
    return PhoneValidationResult(
        valid=False,
        reason=REASON_NOT_FOUND,
        raw_input=(message or "").strip(),
        normalized_input=result.normalized_input,
    )


def parse_phone(raw: str, default_region: str | None = None) -> tuple[str, str] | None:
    result = validate_phone_number(raw, default_region)
    if not result.valid:
        return None
    return result.e164, result.country


def apply_phone_to_state(state: Any, result: PhoneValidationResult) -> None:
    if not result.valid:
        return
    state.phone = result.e164
    state.phone_country = result.country
    state.country = result.country
    if not getattr(state, "city_country", ""):
        state.city_country = CITY_COUNTRY_INDIA
    state.pending_phone = ""


def ingest_phone_message(state: Any, message: str) -> PhoneValidationResult | None:
    awaiting = getattr(state, "awaiting_field", "")
    if getattr(state, "phone", "") and awaiting not in {"phone", "phone_country"}:
        return None
    result = extract_and_validate_phone(message)
    if result.valid:
        apply_phone_to_state(state, result)
        return result
    if awaiting in {"phone", "phone_country"} or looks_like_phone_attempt(message):
        return result
    return None


def phone_validation_reply(result: PhoneValidationResult | dict[str, Any] | str) -> str:
    if isinstance(result, PhoneValidationResult):
        reason = result.reason
    elif isinstance(result, dict):
        reason = str(result.get("reason") or result.get("validation_reason") or "")
    else:
        reason = str(result or "")
    if reason == REASON_INVALID_PREFIX:
        return (
            "That doesn't look like a valid Indian mobile number. "
            "Please enter a 10-digit mobile number starting with 6, 7, 8, or 9."
        )
    return (
        "That number doesn't look like a valid 10-digit mobile number. "
        "Could you please share your 10-digit mobile number?"
    )
