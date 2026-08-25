from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.helpers.conversation_history import compact_recent_history
from app.helpers.conversation_extract import looks_like_contact_request
from app.helpers.generation_json import extract_json_object
from app.helpers.query_normalize import looks_like_knowledge_request
from app.services.conversation.models import (
    ChatMode,
    ConversationGoal,
    ConversationState,
    LeadStage,
    LeadStatus,
    TurnIntent,
)
from app.services.conversation.query_rewriter import QueryRewriter

CONVERSATION_STATUSES = {"SALE", "KNOWLEDGE", "SUPPORT", "OTHER", "NONE"}
CURRENT_STATUSES = {"LEAD", "KNOWLEDGE", "SUPPORT", "GENERAL"}
LEAD_STAGES = {"NOT_STARTED", "IN_PROGRESS", "COMPLETED"}
NEXT_ACTIONS = {
    "ANSWER_KNOWLEDGE",
    "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD",
    "CONTINUE_LEAD",
    "GENERAL_RESPONSE",
    "SALES_PITCH_AND_OFFER_CONTACT",
    "OTHER",
    # legacy aliases accepted from older prompts
    "START_LEAD",
    "RESUME_LEAD",
    "ANSWER_AND_RESUME_LEAD",
    "HANDLE_SUPPORT",
}

STATE_MANAGER_SYSTEM = """You are the Conversation State Manager for a hearing-aid sales and customer-support chatbot.

Your job is to understand the user's CURRENT message using existing conversation state and recent history.

You are NOT the final response generator. Return ONLY valid JSON. No markdown. No explanation.

Core rules:
- conversation_status = persistent business journey (SALE, KNOWLEDGE, SUPPORT, OTHER)
- current_status = what to handle RIGHT NOW (LEAD, KNOWLEDGE, SUPPORT, GENERAL)
- GENERAL is ONLY for greetings, thanks, acknowledgements, small talk, and conversational pleasantries
- Factual, informational, product, company, policy, warranty, pricing, or person-related questions => current_status=KNOWLEDGE, next_action=ANSWER_KNOWLEDGE. Never treat those as GENERAL
- Temporary KNOWLEDGE must NOT erase SALE or SUPPORT
- Answer what the user is asking NOW; never continue an old workflow blindly
- Purchase intent ("I want to buy TINY") => conversation_status=SALE, current_status=LEAD, but do NOT start a contact form
- Lead collection starts only on clear contact/sales action ("contact me", "call me", "let's go ahead")
- Name, phone, city are data — extract when provided, never ask again in your output
- Resolve pronouns using history and active_product before sub_questions
- Put fully self-contained knowledge queries in sub_questions
- diverge=true only when multiple independent knowledge questions exist in the current message
- Do NOT treat purchase intent as a sub-question
- Never mark lead.status=COMPLETED unless tool already succeeded (application is authoritative)
- Do not invent price, warranty, specs, or company facts

Output schema (exact keys only):
{
  "conversation_status": "SALE | KNOWLEDGE | SUPPORT | OTHER",
  "current_status": "LEAD | KNOWLEDGE | SUPPORT | GENERAL",
  "active_product": "string | null",
  "diverge": true,
  "sub_questions": [],
  "sales_interest": true,
  "lead": {
    "status": "NOT_STARTED | IN_PROGRESS | COMPLETED",
    "collected_fields": {},
    "missing_fields": []
  },
  "next_action": "ANSWER_KNOWLEDGE | ANSWER_KNOWLEDGE_THEN_RESUME_LEAD | CONTINUE_LEAD | GENERAL_RESPONSE | OTHER"
}
"""

MULTI_QUESTION_RE = re.compile(
    r"\b(?:and also|and what|as well as|,.*\?|;\s*|\?\s*.*\?)\b",
    re.IGNORECASE,
)
COMPOUND_AND_WHAT_RE = re.compile(r"\band\s+what\s+(?:is|are)\b", re.IGNORECASE)


@dataclass
class LeadStateInfo:
    status: str = LeadStage.NOT_STARTED
    collected_fields: dict[str, str] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StateManagerResult:
    conversation_status: str = "OTHER"
    current_status: str = TurnIntent.GENERAL
    active_product: str | None = None
    diverge: bool = False
    sub_questions: list[str] = field(default_factory=list)
    sales_interest: bool = False
    lead: LeadStateInfo = field(default_factory=LeadStateInfo)
    next_action: str = "OTHER"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lead"] = self.lead.to_dict()
        return payload

    @property
    def lead_stage(self) -> str:
        return self.lead.status


def build_state_manager_prompt(state: ConversationState) -> str:
    collected: dict[str, str] = {}
    missing: list[str] = []
    if state.user_name:
        collected["name"] = state.user_name
    else:
        missing.append("name")
    if state.phone:
        collected["phone"] = "[present]"
    else:
        missing.append("phone")
    if state.city:
        collected["city"] = state.city
    else:
        missing.append("city")
    payload = {
        "conversation_status": conversation_status_label(state),
        "active_product": state.product or None,
        "support_issue": state.support_issue or None,
        "sales_interest": bool(state.sales_interest or state.lead_intent),
        "lead_collection_active": bool(state.lead_collection_active),
        "lead": {
            "status": state.lead_stage or LeadStage.NOT_STARTED,
            "collected_fields": collected,
            "missing_fields": missing,
            "awaiting_field": state.awaiting_field or None,
        },
        "user_context_keys": sorted((state.user_context or {}).keys()),
        "recent_history": compact_recent_history(state.conversation_history),
        "current_message": state.user_message or "",
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_state_manager_result(raw: str) -> StateManagerResult | None:
    payload = extract_json_object(raw)
    if not payload:
        return None
    conversation_status = str(payload.get("conversation_status") or "OTHER").upper()
    if conversation_status not in CONVERSATION_STATUSES:
        return None
    current_status = str(payload.get("current_status") or "GENERAL").upper()
    if current_status not in CURRENT_STATUSES:
        return None
    next_action = _normalize_next_action(str(payload.get("next_action") or "OTHER").upper())
    sub_questions = payload.get("sub_questions") or []
    if not isinstance(sub_questions, list):
        sub_questions = []
    lead_payload = payload.get("lead") or {}
    if not isinstance(lead_payload, dict):
        lead_payload = {}
    lead_status = str(lead_payload.get("status") or LeadStage.NOT_STARTED).upper()
    if lead_status not in LEAD_STAGES:
        lead_status = LeadStage.NOT_STARTED
    collected = lead_payload.get("collected_fields") or {}
    if not isinstance(collected, dict):
        collected = {}
    missing = lead_payload.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = []
    product = payload.get("active_product")
    if product is None:
        product = payload.get("product")
    product_text = str(product).strip() if product else None
    try:
        confidence = float(payload.get("confidence") or 0.82)
    except (TypeError, ValueError):
        confidence = 0.82
    return StateManagerResult(
        conversation_status=conversation_status,
        current_status=current_status,
        active_product=product_text or None,
        diverge=bool(payload.get("diverge")),
        sub_questions=[str(item).strip() for item in sub_questions if str(item).strip()],
        sales_interest=bool(payload.get("sales_interest")),
        lead=LeadStateInfo(
            status=lead_status,
            collected_fields={str(k): str(v) for k, v in collected.items() if v},
            missing_fields=[str(item).strip().lower() for item in missing if str(item).strip()],
        ),
        next_action=next_action,
        confidence=max(0.0, min(1.0, confidence)),
    )


def conversation_goal_from_status(status: str) -> str:
    if status == "SALE":
        return ConversationGoal.SALES
    if status == "SUPPORT":
        return ConversationGoal.SUPPORT
    if status == "KNOWLEDGE":
        return ConversationGoal.KNOWLEDGE
    return ConversationGoal.NONE


def turn_intent_from_status(status: str) -> str:
    mapping = {
        "KNOWLEDGE": TurnIntent.KNOWLEDGE,
        "LEAD": TurnIntent.LEAD_INTENT,
        "SUPPORT": TurnIntent.SUPPORT_INTENT,
        "GENERAL": TurnIntent.GENERAL,
    }
    return mapping.get(status, TurnIntent.UNKNOWN)


def conversation_status_label(state: ConversationState) -> str:
    if state.conversation_goal in {ConversationGoal.LEAD, ConversationGoal.SALES}:
        return "SALE"
    if state.conversation_goal == ConversationGoal.SUPPORT:
        return "SUPPORT"
    if state.conversation_goal == ConversationGoal.KNOWLEDGE:
        return "KNOWLEDGE"
    return "OTHER"


def current_status_label(state: ConversationState) -> str:
    intent = state.current_turn_intent or ""
    trace = state.trace or {}
    if intent == TurnIntent.KNOWLEDGE or (
        intent == TurnIntent.MIXED and bool(trace.get("should_retrieve"))
    ):
        return "KNOWLEDGE"
    if intent in {TurnIntent.LEAD_INTENT, TurnIntent.SALES, TurnIntent.ACTION}:
        return "LEAD"
    if intent in {TurnIntent.SUPPORT_INTENT}:
        return "SUPPORT"
    if intent in {TurnIntent.PROVIDE_INFORMATION, TurnIntent.CONTEXT_UPDATE} and (
        state.lead_collection_active or state.awaiting_field
    ):
        return "LEAD"
    return "GENERAL"


def lead_collected_fields(state: ConversationState) -> dict[str, str]:
    collected: dict[str, str] = {}
    if state.user_name:
        collected["name"] = state.user_name
    if state.phone:
        collected["phone"] = state.phone
    if state.city:
        collected["city"] = state.city
    return collected


def lead_missing_fields(state: ConversationState) -> list[str]:
    missing: list[str] = []
    if not state.user_name:
        missing.append("name")
    if not state.phone:
        missing.append("phone")
    if not state.city:
        missing.append("city")
    return missing


def infer_next_action(state: ConversationState, *, needs_rag: bool) -> str:
    if needs_rag:
        if state.lead_collection_active or state.awaiting_field or state.explicit_action == "create_lead":
            return "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD"
        if (
            state.sales_interest
            and state.current_turn_intent in {TurnIntent.LEAD_INTENT, TurnIntent.SALES}
            and not state.lead_collection_active
        ):
            return "SALES_PITCH_AND_OFFER_CONTACT"
        return "ANSWER_KNOWLEDGE"
    if state.lead_collection_active or state.explicit_action == "create_lead":
        return "CONTINUE_LEAD"
    if (
        state.sales_interest
        and state.current_turn_intent in {TurnIntent.LEAD_INTENT, TurnIntent.SALES}
        and not state.lead_collection_active
    ):
        return "SALES_PITCH_AND_OFFER_CONTACT"
    if state.current_turn_intent == TurnIntent.GENERAL:
        return "GENERAL_RESPONSE"
    return "OTHER"


def resolve_knowledge_queries(state: ConversationState) -> tuple[bool, list[str]]:
    message = (state.user_message or "").strip()
    if not message:
        return False, []
    trace = dict(state.trace or {})
    existing = [str(item).strip() for item in (trace.get("sub_questions") or []) if str(item).strip()]
    if existing:
        return bool(trace.get("diverge")), existing
    rewriter = QueryRewriter()
    scratch = ConversationState.from_dict(state.to_dict())
    rewriter.apply(scratch)
    resolved = (scratch.query_rewritten or message).strip()
    diverge = bool(
        (MULTI_QUESTION_RE.search(message) and message.count("?") > 1)
        or COMPOUND_AND_WHAT_RE.search(message)
    )
    if diverge:
        parts = _split_knowledge_questions(message, state.product or "")
        if len(parts) > 1:
            return True, parts
    return False, [resolved] if resolved else []


def populate_deterministic_state_trace(state: ConversationState, *, needs_rag: bool) -> None:
    trace = dict(state.trace or {})
    trace["conversation_status"] = conversation_status_label(state)
    trace["current_status"] = current_status_label(state)
    trace["active_product"] = state.product or ""
    trace["lead_status"] = state.lead_stage or LeadStage.NOT_STARTED
    trace["lead_collection_active"] = bool(state.lead_collection_active)
    trace["history_turn_count"] = len(state.conversation_history or [])
    if needs_rag:
        diverge, sub_questions = resolve_knowledge_queries(state)
        trace["diverge"] = diverge
        trace["sub_questions"] = sub_questions
        if sub_questions:
            trace["resolved_query"] = sub_questions[0]
    else:
        trace.setdefault("diverge", False)
        trace.setdefault("sub_questions", [])
    trace["next_action"] = infer_next_action(state, needs_rag=needs_rag)
    if trace["next_action"] == "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD":
        trace["resume_lead_after_knowledge"] = True
    state.trace = trace


def apply_state_manager_result(state: ConversationState, result: StateManagerResult) -> None:
    goal = conversation_goal_from_status(result.conversation_status)
    if result.conversation_status != "OTHER" or not state.conversation_goal:
        state.conversation_goal = goal
    if result.conversation_status == "SALE":
        state.sales_interest = True
        state.lead_intent = True
    elif result.conversation_status == "SUPPORT":
        state.support_intent = True
    if result.active_product:
        state.product = result.active_product
    if result.sales_interest:
        state.sales_interest = True
    if state.lead_status != LeadStatus.CREATED:
        state.lead_stage = result.lead.status
    for key, value in result.lead.collected_fields.items():
        if key == "name" and value and not state.user_name:
            state.user_name = value
        elif key == "city" and value and not state.city:
            state.city = value
    state.trace = dict(state.trace or {})
    state.trace["conversation_status"] = result.conversation_status
    state.trace["current_status"] = result.current_status
    state.trace["active_product"] = result.active_product or state.product or ""
    state.trace["diverge"] = result.diverge
    state.trace["sub_questions"] = list(result.sub_questions)
    state.trace["next_action"] = result.next_action
    state.trace["lead_status"] = state.lead_stage
    if result.sub_questions:
        state.trace["resolved_query"] = result.sub_questions[0]
    if result.next_action in {"CONTINUE_LEAD", "START_LEAD", "RESUME_LEAD"}:
        state.lead_collection_active = True
        if looks_like_contact_request(state.user_message or "") or state.awaiting_field:
            state.explicit_action = "create_lead"
    if result.next_action in {"ANSWER_KNOWLEDGE_THEN_RESUME_LEAD", "ANSWER_AND_RESUME_LEAD", "RESUME_LEAD"}:
        state.trace["resume_lead_after_knowledge"] = True


def _normalize_next_action(action: str) -> str:
    aliases = {
        "START_LEAD": "CONTINUE_LEAD",
        "RESUME_LEAD": "CONTINUE_LEAD",
        "ANSWER_AND_RESUME_LEAD": "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD",
        "HANDLE_SUPPORT": "OTHER",
    }
    action = aliases.get(action, action)
    if action not in NEXT_ACTIONS:
        return "OTHER"
    return action


def _split_knowledge_questions(message: str, product: str) -> list[str]:
    if COMPOUND_AND_WHAT_RE.search(message):
        prefix, suffix = re.split(
            r"\s+and\s+what\s+(?:is|are)\s+",
            message,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        parts = [prefix.strip(" ,;."), suffix.strip(" ,;.")]
        resolved: list[str] = []
        for index, part in enumerate(parts):
            text = part.strip(" ,;.")
            if not text:
                continue
            if index == 0:
                if not text.endswith("?"):
                    text = f"{text}?"
            else:
                text = f"What is {text}"
                if not text.endswith("?"):
                    text = f"{text}?"
            if product and product.lower() not in text.lower():
                rewriter = QueryRewriter()
                scratch = ConversationState(user_message=text, product=product)
                rewriter.apply(scratch)
                text = scratch.query_rewritten or text
            resolved.append(text)
        if len(resolved) > 1:
            return resolved
    parts = re.split(r"\?\s*(?:and|,|;|\s+also\s+)", message, flags=re.IGNORECASE)
    resolved: list[str] = []
    for part in parts:
        text = part.strip(" ,;.")
        if not text:
            continue
        if not text.endswith("?"):
            text = f"{text}?"
        if product and product.lower() not in text.lower():
            rewriter = QueryRewriter()
            scratch = ConversationState(user_message=text, product=product)
            rewriter.apply(scratch)
            text = scratch.query_rewritten or text
        resolved.append(text)
    return resolved


def looks_like_standalone_knowledge(message: str) -> bool:
    return looks_like_knowledge_request(message)
