from __future__ import annotations

import logging

from app.services.conversation.lead_service import prompt_for_lead_field
from app.services.conversation.models import (
    ConversationGoal,
    ConversationState,
    LeadStatus,
    LeadWorkflow,
    SupportWorkflow,
    TicketStatus,
)
from app.services.conversation.support_service import prompt_for_support_field

logger = logging.getLogger(__name__)

RESUME_LEAD_ACTION = "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD"
RESUME_SUPPORT_ACTION = "ANSWER_KNOWLEDGE_THEN_RESUME_SUPPORT"


def next_missing_lead_field(state: ConversationState) -> str:
    if state.awaiting_field == "phone_country" and not state.phone:
        return "phone_country"
    if state.awaiting_field == "phone" and not state.phone:
        return "phone_country" if state.pending_phone else "phone"
    if state.awaiting_field == "name" and not state.user_name:
        return "name"
    if state.awaiting_field == "city" and not state.city:
        return "city"
    if not state.user_name:
        return "name"
    if not state.phone:
        return "phone_country" if state.pending_phone else "phone"
    if not state.city:
        return "city"
    return ""


def next_missing_support_field(state: ConversationState) -> str:
    if not state.user_name:
        return "name"
    if not state.product:
        return "product"
    if not state.phone:
        if state.pending_phone or state.awaiting_field == "phone_country":
            return "phone_country"
        return "phone"
    if not state.support_issue:
        return "issue"
    return ""


def is_active_lead_collection(state: ConversationState) -> bool:
    if state.conversation_goal not in {ConversationGoal.LEAD, ConversationGoal.SALES}:
        return False
    if state.lead_status == LeadStatus.CREATED:
        return False
    if not next_missing_lead_field(state):
        return False
    if state.lead_collection_active or state.awaiting_field:
        return True
    trace = state.trace or {}
    if trace.get("next_action") == RESUME_LEAD_ACTION:
        return True
    if trace.get("resume_lead_after_knowledge"):
        return True
    return state.lead_status == LeadStatus.COLLECTING


def should_resume_lead_after_knowledge(state: ConversationState) -> bool:
    return is_active_lead_collection(state)


def should_resume_support_after_knowledge(state: ConversationState) -> bool:
    if state.conversation_goal != ConversationGoal.SUPPORT:
        return False
    if not (state.awaiting_field or state.explicit_action == "create_ticket"):
        return False
    if not next_missing_support_field(state):
        return False
    trace = state.trace or {}
    if trace.get("resume_support_after_knowledge"):
        return True
    return trace.get("next_action") == RESUME_SUPPORT_ACTION


def _apply_lead_resume_state(state: ConversationState, missing: str) -> None:
    state.lead_collection_active = True
    state.lead_status = LeadStatus.COLLECTING
    state.awaiting_field = missing
    mapping = {
        "phone": LeadWorkflow.COLLECTING_PHONE,
        "phone_country": LeadWorkflow.COLLECTING_PHONE,
        "name": LeadWorkflow.COLLECTING_NAME,
        "city": LeadWorkflow.COLLECTING_CITY,
    }
    state.lead_workflow = mapping.get(missing, LeadWorkflow.COLLECTING_PHONE)
    trace = dict(state.trace or {})
    trace["lead_resume_appended"] = True
    trace["ask_missing"] = True
    trace["next_missing_field"] = missing
    state.trace = trace


def _apply_support_resume_state(state: ConversationState, missing: str) -> None:
    state.ticket_status = TicketStatus.COLLECTING
    state.awaiting_field = missing
    mapping = {
        "name": SupportWorkflow.COLLECTING_NAME,
        "product": SupportWorkflow.COLLECTING_PRODUCT,
        "phone": SupportWorkflow.COLLECTING_PHONE,
        "issue": SupportWorkflow.COLLECTING_ISSUE,
    }
    state.support_workflow = mapping.get(missing, SupportWorkflow.UNDERSTANDING_ISSUE)
    trace = dict(state.trace or {})
    trace["support_resume_appended"] = True
    trace["ask_missing"] = True
    trace["next_missing_field"] = missing
    state.trace = trace


def append_workflow_resume_after_knowledge(state: ConversationState, knowledge_answer: str) -> str:
    answer = (knowledge_answer or "").strip()
    if not answer:
        return answer

    next_lead_field = next_missing_lead_field(state)
    next_support_field = next_missing_support_field(state)
    resume_lead = should_resume_lead_after_knowledge(state)
    resume_support = should_resume_support_after_knowledge(state)

    logger.info(
        "knowledge_resume_check conversation_goal=%s lead_collection_active=%s awaiting_field=%s "
        "next_lead_field=%s resume_lead=%s resume_support=%s next_action=%s",
        state.conversation_goal,
        state.lead_collection_active,
        state.awaiting_field,
        next_lead_field,
        resume_lead,
        resume_support,
        (state.trace or {}).get("next_action"),
    )

    if resume_lead and next_lead_field:
        _apply_lead_resume_state(state, next_lead_field)
        continuation = prompt_for_lead_field(
            next_lead_field,
            name=state.user_name,
            product=state.product,
            after_knowledge=True,
        )
        composed = f"{answer}\n\n{continuation}"
        logger.info(
            "knowledge_resume_applied workflow=lead next_missing_field=%s resumed_lead=true",
            next_lead_field,
        )
        return composed

    if resume_support and next_support_field:
        _apply_support_resume_state(state, next_support_field)
        continuation = prompt_for_support_field(next_support_field, name=state.user_name)
        composed = f"{answer}\n\n{continuation}"
        logger.info(
            "knowledge_resume_applied workflow=support next_missing_field=%s resumed_support=true",
            next_support_field,
        )
        return composed

    logger.info("knowledge_resume_skipped resumed_lead=false resumed_support=false")
    return answer
