from __future__ import annotations

import time

from app.services.conversation.models import ConversationState, LeadStatus, LeadWorkflow
from app.services.conversation.query_rewriter import apply_named_product, extract_product
from app.interfaces.providers.business import LeadTool
from app.domain.entities import Lead
from app.helpers.conversation_extract import extract_city, extract_name, extract_phone_from_text
from app.helpers.conversation_reply import lead_created_reply

LEAD_CREATED_MESSAGE = lead_created_reply()


def prompt_for_lead_field(field: str, *, name: str = "") -> str:
    if field == "phone":
        if name:
            return f"Absolutely, {name}. What's the best number for our team to reach you on?"
        return "What's the best number for our team to reach you on?"
    if field == "name":
        return "May I have your name?"
    if field == "city":
        if name:
            return f"Thanks, {name}. Which city should I note for the team?"
        return "Which city should I note for the team?"
    if field == "product":
        return "Which product are you interested in?"
    return ""


class LeadService:
    def __init__(self, tool: LeadTool) -> None:
        self._tool = tool

    def handle(self, state: ConversationState) -> ConversationState:
        if state.lead_status == LeadStatus.CREATED:
            state.response = lead_created_reply(state.user_name)
            state.lead_workflow = LeadWorkflow.CREATED
            return state
        apply_named_product(state)
        self._ingest_fields(state)
        missing = self._next_missing(state)
        self._sync_workflow(state, missing)
        if missing:
            state.lead_status = LeadStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_lead_field(missing, name=state.user_name)
            return state
        try:
            lead_id = self._tool.create_lead(_lead_from_state(state))
        except Exception:
            state.lead_status = LeadStatus.FAILED
            state.error = "lead_create_failed"
            state.response = "I could not create the lead right now. Please try again shortly."
            return state
        state.lead_status = LeadStatus.CREATED
        state.lead_workflow = LeadWorkflow.CREATED
        state.awaiting_field = ""
        state.response = lead_created_reply(state.user_name)
        _mark_lead_created(state, lead_id)
        return state

    def converse(self, state: ConversationState) -> ConversationState:
        if state.lead_status == LeadStatus.CREATED:
            state.response = lead_created_reply(state.user_name)
            state.lead_workflow = LeadWorkflow.CREATED
            return state
        missing = self._prepare(state)
        collect = self._should_collect(state, missing)
        if collect and missing:
            state.lead_status = LeadStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_lead_field(missing, name=state.user_name)
            state.trace["ask_missing"] = True
            _add_capability(state, "LEAD_INFORMATION_COLLECTION")
            return state
        if collect and not missing:
            return self._create(state)
        state.lead_status = LeadStatus.COLLECTING
        if missing:
            state.lead_workflow = LeadWorkflow.DISCUSSING_PRODUCT
            state.awaiting_field = ""
        state.response = ""
        state.trace["needs_natural_reply"] = True
        return state

    def _prepare(self, state: ConversationState) -> str:
        apply_named_product(state)
        self._ingest_fields(state)
        missing = self._next_missing(state)
        self._sync_workflow(state, missing)
        return missing

    def _create(self, state: ConversationState) -> ConversationState:
        started = time.perf_counter()
        try:
            lead_id = self._tool.create_lead(_lead_from_state(state))
        except Exception:
            state.lead_status = LeadStatus.FAILED
            state.error = "lead_create_failed"
            state.response = "I could not create the lead right now. Please try again shortly."
            _record_tool_ms(state, started)
            return state
        state.lead_status = LeadStatus.CREATED
        state.lead_workflow = LeadWorkflow.CREATED
        state.awaiting_field = ""
        state.response = lead_created_reply(state.user_name)
        _mark_lead_created(state, lead_id)
        _record_tool_ms(state, started)
        return state

    def _should_collect(self, state: ConversationState, missing: str) -> bool:
        if state.explicit_action == "create_lead":
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
        city = extract_city(message)
        if city and not state.city:
            state.city = city
        field = state.awaiting_field
        if field == "name" and not state.user_name and message and not parsed:
            if not extract_product(message):
                state.user_name = message
        elif field == "city" and not state.city and message and not parsed:
            state.city = message
        elif field == "product" and message:
            state.product = extract_product(message) or state.product or message

    def _next_missing(self, state: ConversationState) -> str:
        if state.awaiting_field == "phone" and not state.phone:
            return "phone"
        if state.awaiting_field == "name" and not state.user_name:
            return "name"
        if state.awaiting_field == "city" and not state.city:
            return "city"
        if not state.user_name:
            return "name"
        if not state.phone:
            return "phone"
        if not state.city:
            return "city"
        return ""

    def _sync_workflow(self, state: ConversationState, missing: str) -> None:
        if state.lead_status == LeadStatus.CREATED:
            state.lead_workflow = LeadWorkflow.CREATED
            return
        mapping = {
            "phone": LeadWorkflow.COLLECTING_PHONE,
            "name": LeadWorkflow.COLLECTING_NAME,
            "city": LeadWorkflow.COLLECTING_CITY,
            "": LeadWorkflow.READY_TO_CREATE,
        }
        state.lead_workflow = mapping.get(missing, LeadWorkflow.NONE)

    def _prompt(self, field: str, *, name: str = "") -> str:
        return prompt_for_lead_field(field, name=name)


def _lead_from_state(state: ConversationState) -> Lead:
    return Lead(
        name=state.user_name,
        phone=state.phone,
        city=state.city,
        country=state.country,
        product=state.product,
        conversation_id=state.conversation_id,
    )


def _mark_lead_created(state: ConversationState, lead_id: str) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_called"] = "create_lead"
    state.trace["should_create_lead"] = True
    state.trace["lead_id"] = lead_id


def _add_capability(state: ConversationState, capability: str) -> None:
    state.trace = dict(state.trace or {})
    caps = list(state.trace.get("capabilities") or [])
    if capability not in caps:
        caps.append(capability)
    state.trace["capabilities"] = [item for item in caps if item != "CONVERSATION_ONLY"]


def _record_tool_ms(state: ConversationState, started: float) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_ms"] = round((time.perf_counter() - started) * 1000, 3)
