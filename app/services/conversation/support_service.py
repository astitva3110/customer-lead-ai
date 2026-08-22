from __future__ import annotations

import time

from app.services.conversation.models import ConversationState, SupportWorkflow, TicketStatus
from app.services.conversation.query_rewriter import apply_named_product, extract_product
from app.interfaces.providers.business import TicketTool
from app.domain.entities import SupportTicket
from app.helpers.conversation_extract import (
    extract_issue,
    extract_name,
    extract_phone_from_text,
    looks_like_issue,
)
from app.helpers.conversation_reply import ticket_created_reply

TICKET_CREATED_MESSAGE = ticket_created_reply()


def prompt_for_support_field(field: str, *, name: str = "") -> str:
    if field == "name":
        return "Who should I put this ticket under?"
    if field == "product":
        return "Which hearing aid is this about?"
    if field == "phone":
        if name:
            return f"Thanks, {name}. What's the best number for our support team to reach you on?"
        return "What's the best number for our support team to reach you on?"
    if field == "issue":
        return "Tell me a little about what's happening, and I'll see how I can help."
    return ""


class SupportService:
    def __init__(self, tool: TicketTool) -> None:
        self._tool = tool

    def handle(self, state: ConversationState) -> ConversationState:
        if state.ticket_status == TicketStatus.CREATED:
            state.response = ticket_created_reply(state.user_name)
            state.support_workflow = SupportWorkflow.CREATED
            return state
        apply_named_product(state)
        if not state.support_issue:
            issue = extract_issue(state.user_message)
            if issue:
                state.support_issue = issue
        self._ingest_fields(state)
        missing = self._next_missing(state)
        self._sync_workflow(state, missing)
        if missing:
            state.ticket_status = TicketStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_support_field(missing, name=state.user_name)
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
        missing = self._prepare(state)
        collect = self._should_collect(state)
        if collect and missing:
            state.ticket_status = TicketStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_support_field(missing, name=state.user_name)
            state.trace["ask_missing"] = True
            _add_capability(state, "SUPPORT_INFORMATION_COLLECTION")
            return state
        if collect and not missing:
            return self._create(state)
        state.ticket_status = TicketStatus.COLLECTING
        state.support_workflow = SupportWorkflow.UNDERSTANDING_ISSUE
        if not collect:
            state.awaiting_field = ""
        state.response = ""
        state.trace["needs_natural_reply"] = True
        return state

    def _prepare(self, state: ConversationState) -> str:
        apply_named_product(state)
        issue = extract_issue(state.user_message)
        lowered = (state.user_message or "").lower()
        if issue and "tried the troubleshooting" not in lowered:
            if not state.support_issue:
                state.support_issue = issue
            elif issue not in state.support_issue:
                state.support_issue = f"{state.support_issue}; {issue}"
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

    def _should_collect(self, state: ConversationState) -> bool:
        if state.explicit_action == "create_ticket":
            return True
        if state.awaiting_field:
            return True
        return False

    def _ingest_fields(self, state: ConversationState) -> None:
        message = (state.user_message or "").strip()
        parsed = extract_phone_from_text(message)
        if parsed and not state.phone:
            state.phone, state.country = parsed
        name = extract_name(message)
        if name and not state.user_name:
            state.user_name = name
        if not state.product:
            named = extract_product(message)
            if named:
                state.product = named
        field = state.awaiting_field
        if field == "name" and not state.user_name and message and not parsed:
            if not extract_product(message) and not looks_like_issue(message):
                state.user_name = message
        elif field == "product" and message:
            state.product = extract_product(message) or message
        elif field == "issue" and message and not state.support_issue:
            state.support_issue = message

    def _next_missing(self, state: ConversationState) -> str:
        if not state.user_name:
            return "name"
        if not state.product:
            return "product"
        if not state.phone:
            return "phone"
        if not state.support_issue:
            return "issue"
        return ""

    def _sync_workflow(self, state: ConversationState, missing: str) -> None:
        if state.ticket_status == TicketStatus.CREATED:
            state.support_workflow = SupportWorkflow.CREATED
            return
        mapping = {
            "name": SupportWorkflow.COLLECTING_NAME,
            "product": SupportWorkflow.COLLECTING_PRODUCT,
            "phone": SupportWorkflow.COLLECTING_PHONE,
            "issue": SupportWorkflow.COLLECTING_ISSUE,
            "": SupportWorkflow.READY_TO_CREATE,
        }
        state.support_workflow = mapping.get(missing, SupportWorkflow.UNDERSTANDING_ISSUE)

    def _prompt(self, field: str, *, name: str = "") -> str:
        return prompt_for_support_field(field, name=name)


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


def _add_capability(state: ConversationState, capability: str) -> None:
    state.trace = dict(state.trace or {})
    caps = list(state.trace.get("capabilities") or [])
    if capability not in caps:
        caps.append(capability)
    state.trace["capabilities"] = [item for item in caps if item != "CONVERSATION_ONLY"]


def _looks_like_issue(message: str) -> bool:
    return looks_like_issue(message)


def _record_tool_ms(state: ConversationState, started: float) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_ms"] = round((time.perf_counter() - started) * 1000, 3)
