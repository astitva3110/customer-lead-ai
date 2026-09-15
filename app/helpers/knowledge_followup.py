from __future__ import annotations

"""Legacy follow-up id resolution for older interactive button payloads."""

_LEGACY_FOLLOWUPS: dict[str, str] = {
    "followup_return": "What is the return policy?",
    "followup_refund": "How do refunds work?",
    "followup_privacy": "What is the privacy policy?",
    "followup_warranty_period": "What is the warranty period?",
    "followup_warranty_cover": "What does the warranty cover?",
    "followup_warranty_claim": "How do I claim warranty?",
    "followup_prod_warranty": "What is the warranty?",
    "followup_prod_price": "What is the price?",
    "followup_prod_battery": "What is the battery life?",
    "followup_prod_features": "What are the features?",
    "followup_products": "What products does Earkart offer?",
    "followup_support": "How can I get support?",
    "followup_warranty": "What is the warranty policy?",
}


def resolve_followup_message(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text in _LEGACY_FOLLOWUPS:
        return _LEGACY_FOLLOWUPS[text]
    if text.startswith("followup:"):
        return text.split(":", 1)[1].strip()
    return ""
