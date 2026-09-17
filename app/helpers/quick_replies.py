from __future__ import annotations

import re

from app.helpers.conversation_extract import looks_like_hearing_consultation_need
from app.helpers.query_normalize import looks_like_knowledge_request
from app.services.conversation.models import ChatMode, ConversationGoal, ConversationState
from app.services.conversation.query_rewriter import extract_product, is_catalog_scope_query

LEAD_YES = "lead_yes"
LEAD_NO = "lead_no"
TICKET_YES = "ticket_yes"
TICKET_NO = "ticket_no"
BUY_YES = "buy_yes"
BUY_NO = "buy_no"

PENDING_LEAD_OFFER = "lead_offer"
PENDING_TICKET_OFFER = "ticket_offer"
PENDING_PRODUCT_INTEREST = "product_interest"

_POLICY_QUERY_RE = re.compile(
    r"\b(?:terms|conditions|privacy|refund|return policy|cookie)\b",
    re.IGNORECASE,
)
_HEARING_AID_RE = re.compile(r"\bhearing aids?\b", re.IGNORECASE)


def yes_no_replies(*, yes_id: str, no_id: str) -> list[dict[str, str]]:
    return [
        {"id": yes_id, "label": "Yes"},
        {"id": no_id, "label": "No"},
    ]


def set_trace_quick_replies(
    state: ConversationState,
    replies: list[dict[str, str]],
    *,
    pending_choice: str = "",
) -> None:
    trace = dict(state.trace or {})
    trace["quick_replies"] = replies
    trace["pending_choice"] = pending_choice or ""
    state.trace = trace


def clear_trace_quick_replies(state: ConversationState) -> None:
    trace = dict(state.trace or {})
    trace.pop("quick_replies", None)
    trace.pop("pending_choice", None)
    state.trace = trace


def declined_choice_reply() -> str:
    return "No worries at all. How else can I help you today?"


def lead_choice_prompt(product: str = "") -> str:
    label = (product or "").strip() or "our products"
    return f"Great choice on {label}! Would you like me to connect you with our team?"


CONSULTATION_TRIAL_MESSAGE = "Please create a trial for me"
CONSULTATION_TRIAL_LABEL = "Create trial for me"


def product_interest_message(product: str = "") -> str:
    name = (product or "").strip()
    if name:
        return f"I want {name}"
    return "I want a hearing aid"


def consultation_trial_message() -> str:
    return CONSULTATION_TRIAL_MESSAGE


def _truncate_label(text: str, limit: int = 20) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def build_product_interest_replies(product: str = "", *, include_trial: bool = False) -> list[dict[str, str]]:
    replies: list[dict[str, str]] = []
    if product:
        message = product_interest_message(product)
        replies.append({"id": f"followup:{message}", "label": _truncate_label(message)})
    elif not include_trial:
        message = product_interest_message("")
        replies.append({"id": f"followup:{message}", "label": _truncate_label(message)})
    if include_trial:
        trial_msg = consultation_trial_message()
        replies.append(
            {
                "id": f"followup:{trial_msg}",
                "label": _truncate_label(CONSULTATION_TRIAL_LABEL),
            }
        )
    if not product and include_trial:
        message = product_interest_message("")
        replies.insert(0, {"id": f"followup:{message}", "label": _truncate_label(message)})
    return replies[:3]


def should_attach_product_interest(state: ConversationState) -> bool:
    trace = state.trace or {}
    if trace.get("pending_choice"):
        return False
    if state.lead_collection_active or state.support_collection_active or state.awaiting_field:
        return False
    if state.conversation_goal == ConversationGoal.SUPPORT or state.support_intent:
        return False
    if state.explicit_action in {"create_lead", "create_ticket"}:
        return False
    if trace.get("price_lead_offer") or trace.get("sales_pitch_from_rag"):
        return False
    message = (state.user_message or "").strip()
    if not message:
        return False
    product = (state.product or extract_product(message) or "").strip()
    hearing_mention = bool(_HEARING_AID_RE.search(message))
    catalog = is_catalog_scope_query(message)
    if _POLICY_QUERY_RE.search(message) and not product and not hearing_mention:
        return False
    if looks_like_hearing_consultation_need(message):
        return True
    if product and (looks_like_knowledge_request(message) or hearing_mention or catalog):
        return True
    return hearing_mention or catalog


def should_include_consultation_trial(state: ConversationState) -> bool:
    message = (state.user_message or "").strip()
    return looks_like_hearing_consultation_need(message)


def attach_product_interest_quick_replies(state: ConversationState) -> None:
    if not should_attach_product_interest(state):
        return
    product = (state.product or extract_product(state.user_message or "") or "").strip()
    include_trial = should_include_consultation_trial(state)
    replies = build_product_interest_replies(product, include_trial=include_trial)
    if not replies:
        return
    set_trace_quick_replies(state, replies, pending_choice=PENDING_PRODUCT_INTEREST)


def buy_choice_prompt(product: str = "") -> str:
    label = (product or "").strip() or "it"
    return f"Would you like to buy {label}?"


def attach_lead_offer_buttons(state: ConversationState, *, product: str = "") -> None:
    """Attach Yes/No when the reply already offers lead/sales contact; else append the offer."""
    if offered_lead_choice(state.response):
        set_trace_quick_replies(
            state,
            yes_no_replies(yes_id=LEAD_YES, no_id=LEAD_NO),
            pending_choice=PENDING_LEAD_OFFER,
        )
        return
    set_lead_choice_offer(state, product=product or state.product)


def attach_lead_offer_after_sales_pitch(state: ConversationState) -> None:
    """Attach Yes/No buttons after buy pitch; avoid duplicating the lead-offer paragraph."""
    trace = state.trace or {}
    if not (
        trace.get("sales_pitch_from_rag")
        or trace.get("next_action") == "SALES_PITCH_AND_OFFER_CONTACT"
    ):
        return
    if state.lead_collection_active or trace.get("lead_resume_appended"):
        return
    attach_lead_offer_buttons(state, product=state.product)


def set_lead_choice_offer(state: ConversationState, *, product: str = "") -> None:
    suffix = lead_choice_prompt(product or state.product)
    _append_suffix(state, suffix)
    set_trace_quick_replies(
        state,
        yes_no_replies(yes_id=LEAD_YES, no_id=LEAD_NO),
        pending_choice=PENDING_LEAD_OFFER,
    )


def set_ticket_choice_offer(state: ConversationState) -> None:
    set_trace_quick_replies(
        state,
        yes_no_replies(yes_id=TICKET_YES, no_id=TICKET_NO),
        pending_choice=PENDING_TICKET_OFFER,
    )


def resolve_inbound_choice(message: str) -> dict[str, str] | None:
    from app.helpers.knowledge_followup import resolve_followup_message

    text = (message or "").strip()
    if not text:
        return None
    lowered = text.lower()
    mapping = {
        LEAD_YES: {"action": "lead_yes"},
        LEAD_NO: {"action": "lead_no"},
        TICKET_YES: {"action": "ticket_yes"},
        TICKET_NO: {"action": "ticket_no"},
    }
    if text in mapping:
        return mapping[text]
    if lowered in {"yes", "no"}:
        return {"action": "yes" if lowered == "yes" else "no"}
    followup = resolve_followup_message(text)
    if followup:
        return {"action": "followup", "message": followup}
    return None


def quick_replies_from_trace(state: ConversationState) -> list[dict[str, str]]:
    trace = state.trace or {}
    pending = str(trace.get("pending_choice") or "")
    if pending not in {PENDING_LEAD_OFFER, PENDING_TICKET_OFFER, PENDING_PRODUCT_INTEREST}:
        return []
    items = trace.get("quick_replies") or []
    return [dict(item) for item in items if isinstance(item, dict)]


def apply_declined_choice(state: ConversationState) -> None:
    clear_trace_quick_replies(state)
    state.response = declined_choice_reply()
    trace = dict(state.trace or {})
    trace["needs_natural_reply"] = False
    trace["choice_declined"] = True
    trace.pop("sales_pitch_from_rag", None)
    trace.pop("next_action", None)
    state.trace = trace
    state.lead_collection_active = False
    state.support_collection_active = False
    state.lead_intent = False
    state.sales_interest = False
    state.awaiting_field = ""
    state.explicit_action = ""
    state.mode = ChatMode.KNOWLEDGE
    state.intent = ChatMode.KNOWLEDGE
    if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES}:
        state.conversation_goal = ConversationGoal.KNOWLEDGE


_LEAD_TEAM_OFFER_RE = re.compile(
    r"(?:"
    r"connect(?:\s+you)?\s+with\s+(?:our\s+)?team"
    r"|our team can contact"
    r"|connect you with our sales team"
    r"|sales team can contact"
    r")",
    re.IGNORECASE,
)


def offered_lead_choice(assistant_text: str) -> bool:
    return bool(_LEAD_TEAM_OFFER_RE.search(assistant_text or ""))


def offered_ticket_choice(assistant_text: str) -> bool:
    return bool(
        re.search(
            r"connect you with our team|our team can reach|connect you with our customer service team|customer service team can reach",
            assistant_text or "",
            flags=re.IGNORECASE,
        )
    )


def _append_suffix(state: ConversationState, suffix: str) -> None:
    body = (state.response or "").strip()
    extra = (suffix or "").strip()
    if not extra:
        return
    if extra.lower() in body.lower():
        return
    state.response = f"{body}\n\n{extra}" if body else extra
