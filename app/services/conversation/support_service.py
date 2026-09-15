from __future__ import annotations

import logging
import time

from app.domain.entities import SupportTicket
from app.helpers.conversation_extract import (
    extract_issue,
    extract_name,
    looks_like_issue,
    looks_like_support_contact_request,
    looks_like_support_escalation_request,
    looks_like_ticket_request,
    normalize_person_name,
)
from app.helpers.conversation_reply import ticket_created_reply
from app.helpers.conversation_turn import is_acknowledgement_only
from app.helpers.phone import (
    PhoneValidationResult,
    ingest_phone_message,
    looks_like_phone_attempt,
    phone_validation_reply,
)
from app.interfaces.providers.business import TicketTool
from app.helpers.conversation_reply import support_ticket_offer_reply
from app.helpers.quick_replies import set_ticket_choice_offer
from app.services.conversation.models import ConversationState, SupportWorkflow, TicketStatus, TurnIntent
from app.services.conversation.query_rewriter import apply_named_product, extract_product

logger = logging.getLogger(__name__)

TICKET_CREATED_MESSAGE = ticket_created_reply()


def prompt_for_support_field(field: str, *, name: str = "") -> str:
    display_name = normalize_person_name(name)
    if field == "phone":
        if display_name:
            return f"Thanks, {display_name}. What's the best number for our team to reach you on?"
        return "What's the best number for our team to reach you on?"
    if field == "name":
        return "What name should our team use when they reach you?"
    if field == "product":
        return "Which hearing aid is this about?"
    if field == "phone_country":
        return "Which country is this number from?"
    if field == "issue":
        return "Tell me a little about what's happening, and I'll see how I can help."
    return ""


def prompt_for_missing_support_field(state: ConversationState, field: str) -> str:
    validation = (state.trace or {}).get("phone_validation") or {}
    if field in {"phone", "phone_country"} and validation.get("valid") is False:
        return phone_validation_reply(validation)
    return prompt_for_support_field(field, name=state.user_name)


class SupportService:
    def __init__(self, tool: TicketTool) -> None:
        self._tool = tool

    def handle(self, state: ConversationState) -> ConversationState:
        if state.ticket_status == TicketStatus.CREATED:
            state.response = ticket_created_reply(state.user_name)
            state.support_workflow = SupportWorkflow.CREATED
            return state
        apply_named_product(state)
        self._capture_issue(state)
        self._ingest_fields(state)
        if _is_phone_retry(state):
            return self._retry_phone(state)
        missing = self._next_missing(state)
        self._sync_workflow(state, missing)
        _log_ticket_completion(state, missing)
        if missing:
            state.ticket_status = TicketStatus.COLLECTING
            state.support_collection_active = True
            state.awaiting_field = missing
            state.response = prompt_for_missing_support_field(state, missing)
            return state
        try:
            ticket_id = self._tool.create_support_ticket(_ticket_from_state(state))
        except Exception:
            state.ticket_status = TicketStatus.FAILED
            state.error = "ticket_create_failed"
            state.response = "I could not create the support ticket right now. Please try again shortly."
            return state
        state.ticket_status = TicketStatus.CREATED
        state.support_workflow = SupportWorkflow.CREATED
        state.awaiting_field = ""
        state.response = ticket_created_reply(state.user_name)
        _mark_ticket_created(state, ticket_id)
        return state

    def converse(self, state: ConversationState) -> ConversationState:
        if state.ticket_status == TicketStatus.CREATED:
            state.response = ticket_created_reply(state.user_name)
            state.support_workflow = SupportWorkflow.CREATED
            return state
        if (
            is_acknowledgement_only(state.user_message or "")
            and state.awaiting_field
            and state.support_collection_active
        ):
            state.ticket_status = TicketStatus.COLLECTING
            state.response = ""
            state.trace["needs_natural_reply"] = True
            state.trace["acknowledgement_only"] = True
            return state
        missing = self._prepare(state)
        _log_ticket_completion(state, missing)
        if _is_phone_retry(state):
            return self._retry_phone(state)
        collect = self._should_collect(state, missing)
        if collect and missing:
            state.support_collection_active = True
            state.ticket_status = TicketStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_missing_support_field(state, missing)
            state.trace["ask_missing"] = True
            _add_capability(state, "SUPPORT_INFORMATION_COLLECTION")
            return state
        if collect and not missing:
            return self._create(state)
        state.ticket_status = TicketStatus.COLLECTING
        if missing:
            state.support_workflow = SupportWorkflow.UNDERSTANDING_ISSUE
            state.awaiting_field = ""
        if (
            state.current_turn_intent == TurnIntent.SUPPORT_INTENT
            and not state.support_collection_active
            and not state.awaiting_field
        ):
            state.response = support_ticket_offer_reply(state)
            set_ticket_choice_offer(state)
            state.trace["needs_natural_reply"] = False
            return state
        state.response = ""
        state.trace["needs_natural_reply"] = True
        return state

    def _prepare(self, state: ConversationState) -> str:
        apply_named_product(state)
        self._capture_issue(state)
        self._ingest_fields(state)
        missing = self._next_missing(state)
        self._sync_workflow(state, missing)
        return missing

    def _create(self, state: ConversationState) -> ConversationState:
        started = time.perf_counter()
        try:
            ticket_id = self._tool.create_support_ticket(_ticket_from_state(state))
        except Exception:
            state.ticket_status = TicketStatus.FAILED
            state.error = "ticket_create_failed"
            state.response = "I could not create the support ticket right now. Please try again shortly."
            _record_tool_ms(state, started)
            return state
        state.ticket_status = TicketStatus.CREATED
        state.support_workflow = SupportWorkflow.CREATED
        state.awaiting_field = ""
        state.response = ticket_created_reply(state.user_name)
        _mark_ticket_created(state, ticket_id)
        _record_tool_ms(state, started)
        return state

    def _should_collect(self, state: ConversationState, missing: str) -> bool:
        if state.explicit_action == "create_ticket":
            return True
        if state.awaiting_field:
            return True
        if state.support_collection_active:
            return True
        return False

    def _capture_issue(self, state: ConversationState) -> None:
        issue = extract_issue(state.user_message)
        lowered = (state.user_message or "").lower()
        if issue and "tried the troubleshooting" not in lowered:
            if not state.support_issue:
                state.support_issue = issue
            elif issue not in state.support_issue:
                state.support_issue = f"{state.support_issue}; {issue}"

    def _ingest_fields(self, state: ConversationState) -> None:
        message = (state.user_message or "").strip()
        previous_field = state.awaiting_field or self._next_missing(state)
        awaiting_phone = previous_field in {"phone", "phone_country"} and not state.phone
        parsed_phone = self._ingest_phone(state, message, previous_field, awaiting_phone)
        if _is_phone_retry(state):
            return
        name = extract_name(message)
        if name and not state.user_name:
            state.user_name = name
        if not state.product:
            named = extract_product(message)
            if named:
                state.product = named
        field = state.awaiting_field
        if field == "name" and not state.user_name and message and not parsed_phone:
            if _accepts_support_name(message):
                state.user_name = normalize_person_name(message)
        elif field == "product" and message:
            state.product = extract_product(message) or state.product or message
        elif field == "issue" and message and not state.support_issue:
            state.support_issue = message

    def _ingest_phone(
        self,
        state: ConversationState,
        message: str,
        previous_field: str,
        awaiting_phone: bool,
    ) -> bool:
        result = ingest_phone_message(state, message)
        should_check = awaiting_phone or looks_like_phone_attempt(message)
        if result is None:
            return False
        if result.valid:
            next_field = "name" if not state.user_name else ("product" if not state.product else "")
            if should_check:
                _record_phone_validation(state, message, result, previous_field, next_field)
            return True
        if awaiting_phone and looks_like_phone_attempt(message):
            _mark_phone_retry(state, result)
            _record_phone_validation(state, message, result, previous_field, "phone")
        return False

    def _retry_phone(self, state: ConversationState) -> ConversationState:
        validation = (state.trace or {}).get("phone_validation") or {}
        state.support_collection_active = True
        state.ticket_status = TicketStatus.COLLECTING
        state.awaiting_field = "phone"
        state.support_workflow = SupportWorkflow.COLLECTING_PHONE
        state.response = phone_validation_reply(validation)
        _add_capability(state, "SUPPORT_INFORMATION_COLLECTION")
        return state

    def _next_missing(self, state: ConversationState) -> str:
        if state.awaiting_field == "phone_country" and not state.phone:
            return "phone_country"
        if state.awaiting_field == "phone" and not state.phone:
            return "phone_country" if state.pending_phone else "phone"
        if state.awaiting_field == "name" and not state.user_name:
            return "name"
        if state.awaiting_field == "product" and not state.product:
            return "product"
        if state.awaiting_field == "issue" and not state.support_issue:
            return "issue"
        if not state.user_name:
            return "name"
        if not state.phone:
            return "phone_country" if state.pending_phone else "phone"
        if not state.product:
            return "product"
        if not state.support_issue:
            return "issue"
        return ""

    def _sync_workflow(self, state: ConversationState, missing: str) -> None:
        if state.ticket_status == TicketStatus.CREATED:
            state.support_workflow = SupportWorkflow.CREATED
            return
        mapping = {
            "phone": SupportWorkflow.COLLECTING_PHONE,
            "phone_country": SupportWorkflow.COLLECTING_PHONE,
            "name": SupportWorkflow.COLLECTING_NAME,
            "product": SupportWorkflow.COLLECTING_PRODUCT,
            "issue": SupportWorkflow.COLLECTING_ISSUE,
            "": SupportWorkflow.READY_TO_CREATE,
        }
        state.support_workflow = mapping.get(missing, SupportWorkflow.UNDERSTANDING_ISSUE)


def _accepts_support_name(message: str) -> bool:
    text = (message or "").strip()
    if not text or len(text.split()) > 4:
        return False
    if looks_like_phone_attempt(text):
        return False
    if extract_product(text):
        return False
    if looks_like_issue(text):
        return False
    if looks_like_ticket_request(text):
        return False
    if looks_like_support_contact_request(text):
        return False
    if looks_like_support_escalation_request(text):
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("already", "tried", "cleaned", "please", "help", "what", "how")):
        return False
    return True


def _ticket_from_state(state: ConversationState) -> SupportTicket:
    return SupportTicket(
        name=state.user_name,
        phone=state.phone,
        product=state.product,
        issue=state.support_issue,
        conversation_id=state.conversation_id,
    )


def _mark_ticket_created(state: ConversationState, ticket_id: str) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_called"] = "create_support_ticket"
    state.trace["should_create_ticket"] = True
    state.trace["ticket_id"] = ticket_id
    logger.info("CREATE_TICKET %s", {"called": True, "ticket_id": ticket_id, "product": state.product})


def _log_ticket_completion(state: ConversationState, missing: str) -> None:
    missing_fields = [
        field
        for field, value in (
            ("name", state.user_name),
            ("phone", state.phone),
            ("product", state.product),
            ("issue", state.support_issue),
        )
        if not value
    ]
    payload = {
        "name": state.user_name or None,
        "phone": bool(state.phone),
        "product": state.product or None,
        "issue": bool(state.support_issue),
        "missing_fields": missing_fields,
        "next_missing_field": missing or None,
        "awaiting_field": state.awaiting_field or None,
        "will_create_ticket": not missing,
    }
    logger.info("TICKET_MISSING %s", payload)
    state.trace = dict(state.trace or {})
    state.trace["ticket_missing"] = payload


def _is_phone_retry(state: ConversationState) -> bool:
    return (state.trace or {}).get("next_action") == "RETRY_PHONE"


def _mark_phone_retry(state: ConversationState, result: PhoneValidationResult) -> None:
    state.trace = dict(state.trace or {})
    state.trace["next_action"] = "RETRY_PHONE"
    state.trace["validation_reason"] = result.reason
    state.trace["phone_retry"] = True


def _record_phone_validation(
    state: ConversationState,
    raw_input: str,
    result: PhoneValidationResult,
    previous_state: str,
    next_state: str,
) -> None:
    payload = result.to_dict()
    payload.update(
        {
            "field": "phone",
            "raw_input": raw_input,
            "previous_state": previous_state,
            "next_state": next_state,
            "validation_reason": result.reason,
        }
    )
    state.trace = dict(state.trace or {})
    state.trace["phone_validation"] = payload
    state.trace["validation_reason"] = result.reason


def _add_capability(state: ConversationState, capability: str) -> None:
    state.trace = dict(state.trace or {})
    caps = list(state.trace.get("capabilities") or [])
    if capability not in caps:
        caps.append(capability)
    state.trace["capabilities"] = [item for item in caps if item != "CONVERSATION_ONLY"]


def _record_tool_ms(state: ConversationState, started: float) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_ms"] = round((time.perf_counter() - started) * 1000, 3)
