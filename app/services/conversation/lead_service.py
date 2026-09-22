from __future__ import annotations

import logging
import time

from app.domain.entities import Lead
from app.helpers.conversation_extract import (
    extract_city,
    extract_name,
    looks_like_purchase_intent,
    normalize_person_name,
)
from app.helpers.conversation_reply import lead_created_reply
from app.helpers.crm_lead import (
    crm_city_value,
    crm_names_from_lead,
    crm_phone_digits,
    crm_problem_from_lead,
    crm_source_for_state,
)
from app.helpers.user_language import localized_text, response_language
from app.helpers.conversation_turn import is_acknowledgement_only
from app.helpers.phone import (
    CITY_COUNTRY_INDIA,
    PhoneValidationResult,
    ingest_phone_message,
    looks_like_phone_attempt,
    phone_validation_reply,
)
from app.interfaces.providers.business import LeadTool
from app.interfaces.providers.crm import CrmLeadPort
from app.services.conversation.models import ConversationState, LeadStage, LeadStatus, LeadWorkflow
from app.services.conversation.query_rewriter import apply_named_product, extract_product

logger = logging.getLogger(__name__)

LEAD_CREATED_MESSAGE = lead_created_reply()


def _prior_user_purchase(state: ConversationState) -> bool:
    current = (state.user_message or "").strip().lower()
    for item in reversed(state.conversation_history or []):
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content or content.lower() == current:
            continue
        return looks_like_purchase_intent(content)
    return False


def prompt_for_lead_field(
    field: str,
    *,
    name: str = "",
    product: str = "",
    after_knowledge: bool = False,
    language: str = "en",
) -> str:
    display_name = normalize_person_name(name)
    if after_knowledge:
        return _knowledge_lead_continuation(field, product=product, language=language)
    if field == "phone":
        key = "ask_lead_phone_named" if display_name else "ask_lead_phone"
        return localized_text(key, language, name=display_name)
    if field == "name":
        return localized_text("ask_lead_name", language)
    if field == "city":
        return localized_text("ask_lead_city", language)
    if field == "phone_country":
        return "Which country is this number from?"
    if field == "product":
        return "Which product are you interested in?"
    return ""


def prompt_for_missing_lead_field(state: ConversationState, field: str) -> str:
    validation = (state.trace or {}).get("phone_validation") or {}
    if field in {"phone", "phone_country"} and validation.get("valid") is False:
        return phone_validation_reply(validation)
    return prompt_for_lead_field(
        field,
        name=state.user_name,
        language=response_language(state),
    )


def _knowledge_lead_continuation(field: str, *, product: str = "", language: str = "en") -> str:
    topic = (product or "").strip() or "this"
    if field == "phone":
        if language == "hi":
            return f"{topic} में मदद के लिए, आपसे संपर्क करने के लिए सबसे अच्छा नंबर क्या है?"
        if language == "hinglish":
            return f"{topic} ke liye help chahiye ho to best contact number kya hai?"
        return f"If you'd like help with {topic}, what's the best number to reach you on?"
    if field == "name":
        if language == "hi":
            return f"{topic} में मदद के लिए, हमारी टीम आपको किस नाम से बुलाए?"
        if language == "hinglish":
            return f"{topic} ke liye help chahiye ho to hamari team aapko kis naam se call kare?"
        return f"If you'd like help with {topic}, what name should our team use?"
    if field == "city":
        if language == "hi":
            return f"{topic} में मदद के लिए, टीम के लिए कौन-सा शहर नोट करूँ?"
        if language == "hinglish":
            return f"{topic} ke liye help chahiye ho to kaunsa city note karoon?"
        return f"If you'd like help with {topic}, which city should I note for the team?"
    if field == "phone_country":
        return "Which country is this number from?"
    return ""


class LeadService:
    def __init__(self, tool: LeadTool, crm: CrmLeadPort | None = None) -> None:
        self._tool = tool
        self._crm = crm

    def handle(self, state: ConversationState) -> ConversationState:
        if state.lead_status == LeadStatus.CREATED:
            return _release_completed_lead(state)
        apply_named_product(state)
        self._ingest_fields(state)
        self._maybe_persist_lead(state)
        if _is_phone_retry(state):
            return self._retry_phone(state)
        missing = self._next_prompt_field(state)
        self._sync_workflow(state, missing)
        _log_lead_completion(state, missing)
        if missing:
            state.lead_status = LeadStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_missing_lead_field(state, missing)
            return state
        if self._is_collection_complete(state):
            return self._complete_lead(state)
        state.lead_status = LeadStatus.COLLECTING
        state.response = ""
        return state

    def converse(self, state: ConversationState) -> ConversationState:
        if state.lead_status == LeadStatus.CREATED:
            return _release_completed_lead(state)
        if (
            is_acknowledgement_only(state.user_message or "")
            and state.awaiting_field
            and state.lead_collection_active
        ):
            state.lead_status = LeadStatus.COLLECTING
            state.response = ""
            state.trace["needs_natural_reply"] = True
            state.trace["acknowledgement_only"] = True
            return state
        missing = self._prepare(state)
        _log_lead_completion(state, missing)
        if _is_phone_retry(state):
            return self._retry_phone(state)
        collect = self._should_collect(state, missing)
        if collect and missing:
            state.lead_collection_active = True
            state.lead_status = LeadStatus.COLLECTING
            state.awaiting_field = missing
            state.response = prompt_for_missing_lead_field(state, missing)
            state.trace["ask_missing"] = True
            _add_capability(state, "LEAD_INFORMATION_COLLECTION")
            return state
        if collect and not missing and self._is_collection_complete(state):
            return self._complete_lead(state)
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
        self._maybe_persist_lead(state)
        missing = self._next_prompt_field(state)
        self._sync_workflow(state, missing)
        return missing

    def _maybe_persist_lead(self, state: ConversationState) -> None:
        if not state.phone:
            return
        trace = dict(state.trace or {})
        if trace.get("lead_id"):
            return
        lead = _lead_from_state(state)
        lead.city = ""
        try:
            lead_id = self._tool.create_lead(lead)
        except Exception:
            logger.exception("lead persist failed")
            return
        trace["lead_id"] = lead_id
        trace["lead_persisted"] = True
        crm_synced = False
        if self._crm is not None:
            try:
                self._crm.create_lead(
                    names=crm_names_from_lead(lead),
                    phone=crm_phone_digits(lead.phone),
                    source=crm_source_for_state(state),
                    problem=crm_problem_from_lead(lead),
                    city=crm_city_value(state.city),
                )
                crm_synced = True
            except Exception:
                logger.exception("crm lead sync failed")
        trace["crm_synced"] = crm_synced
        state.trace = trace
        logger.info(
            "PERSIST_LEAD %s",
            {"lead_id": lead_id, "crm_synced": crm_synced, "city": state.city or None},
        )

    def _complete_lead(self, state: ConversationState) -> ConversationState:
        started = time.perf_counter()
        trace = dict(state.trace or {})
        lead = _lead_from_state(state)
        existing_id = trace.get("lead_id")
        tool_action = "create_lead"
        try:
            if existing_id:
                lead.lead_id = str(existing_id)
                updated = self._tool.update_lead(lead)
                if updated is None:
                    lead_id = self._tool.create_lead(lead)
                else:
                    lead_id = updated.lead_id
                    tool_action = "update_lead"
            else:
                lead_id = self._tool.create_lead(lead)
        except Exception:
            state.lead_status = LeadStatus.FAILED
            state.error = "lead_create_failed"
            state.response = "I could not create the lead right now. Please try again shortly."
            _record_tool_ms(state, started)
            return state
        state.lead_status = LeadStatus.CREATED
        state.lead_workflow = LeadWorkflow.CREATED
        state.lead_stage = LeadStage.COMPLETED
        state.lead_collection_active = False
        state.awaiting_field = ""
        state.explicit_action = ""
        state.response = lead_created_reply(
            state.user_name,
            language=response_language(state),
        )
        _mark_lead_created(state, lead_id, tool_action=tool_action)
        _record_tool_ms(state, started)
        return state

    def _is_collection_complete(self, state: ConversationState) -> bool:
        return bool(state.user_name and state.phone and state.city)

    def _should_collect(self, state: ConversationState, missing: str) -> bool:
        if state.explicit_action == "create_lead":
            return True
        if state.awaiting_field:
            return True
        if state.lead_collection_active:
            return True
        if looks_like_purchase_intent(state.user_message or "") and _prior_user_purchase(state):
            return True
        return False

    def _ingest_fields(self, state: ConversationState) -> None:
        message = (state.user_message or "").strip()
        previous_field = state.awaiting_field or self._next_prompt_field(state)
        awaiting_phone = previous_field in {"phone", "phone_country"} and not state.phone
        parsed_phone = self._ingest_phone(state, message, previous_field, awaiting_phone)
        if _is_phone_retry(state):
            return
        name = extract_name(message)
        if name and not state.user_name:
            state.user_name = name
        city = extract_city(message)
        if city and not state.city:
            state.city = city
            state.city_country = CITY_COUNTRY_INDIA
        field = state.awaiting_field
        if field == "name" and not state.user_name and message and not parsed_phone:
            if not looks_like_phone_attempt(message) and not extract_product(message):
                state.user_name = normalize_person_name(message)
        elif field == "city" and not state.city and message and not parsed_phone:
            if not looks_like_phone_attempt(message):
                state.city = message.strip()
                state.city_country = CITY_COUNTRY_INDIA
        elif field == "product" and message:
            state.product = extract_product(message) or state.product or message
        logger.info(
            "CITY_INGEST %s",
            {
                "raw_input": message,
                "extract_city": city,
                "awaiting_field": field,
                "parsed_phone_this_turn": parsed_phone,
                "city_set": bool(state.city),
                "city": state.city,
            },
        )

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
            next_field = "name" if not state.user_name else ("city" if not state.city else "")
            if should_check:
                _record_phone_validation(state, message, result, previous_field, next_field)
            return True
        if awaiting_phone and looks_like_phone_attempt(message):
            _mark_phone_retry(state, result)
            _record_phone_validation(state, message, result, previous_field, "phone")
        return False

    def _retry_phone(self, state: ConversationState) -> ConversationState:
        validation = (state.trace or {}).get("phone_validation") or {}
        state.lead_collection_active = True
        state.lead_status = LeadStatus.COLLECTING
        state.awaiting_field = "phone"
        state.lead_workflow = LeadWorkflow.COLLECTING_PHONE
        state.response = phone_validation_reply(validation)
        _add_capability(state, "LEAD_INFORMATION_COLLECTION")
        return state

    def _next_prompt_field(self, state: ConversationState) -> str:
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

    def _sync_workflow(self, state: ConversationState, missing: str) -> None:
        if state.lead_status == LeadStatus.CREATED:
            state.lead_workflow = LeadWorkflow.CREATED
            return
        mapping = {
            "phone": LeadWorkflow.COLLECTING_PHONE,
            "phone_country": LeadWorkflow.COLLECTING_PHONE,
            "name": LeadWorkflow.COLLECTING_NAME,
            "city": LeadWorkflow.COLLECTING_CITY,
            "": LeadWorkflow.READY_TO_CREATE,
        }
        state.lead_workflow = mapping.get(missing, LeadWorkflow.NONE)

    def _prompt(self, field: str, *, name: str = "", state: ConversationState | None = None) -> str:
        language = response_language(state) if state is not None else "en"
        return prompt_for_lead_field(field, name=name, language=language)


def _release_completed_lead(state: ConversationState) -> ConversationState:
    """Lead already saved. Do not re-enter collection or repeat the completion line."""
    state.lead_workflow = LeadWorkflow.CREATED
    state.lead_stage = LeadStage.COMPLETED
    state.lead_collection_active = False
    state.awaiting_field = ""
    state.explicit_action = ""
    state.response = ""
    state.trace = dict(state.trace or {})
    state.trace["needs_natural_reply"] = True
    state.trace["lead_workflow_released"] = True
    return state


def _log_lead_completion(state: ConversationState, missing: str) -> None:
    missing_fields = [
        field
        for field, value in (("name", state.user_name), ("phone", state.phone), ("city", state.city))
        if not value
    ]
    payload = {
        "name": state.user_name or None,
        "phone": bool(state.phone),
        "city": state.city or None,
        "missing_fields": missing_fields,
        "next_missing_field": missing or None,
        "awaiting_field": state.awaiting_field or None,
        "will_complete_lead": not missing and bool(state.user_name and state.phone and state.city),
    }
    logger.info("LEAD_MISSING %s", payload)
    state.trace = dict(state.trace or {})
    state.trace["lead_missing"] = payload


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
    logger.info("PHONE_INPUT %s", payload)
    if result.valid:
        logger.info(
            "PHONE_VALIDATION %s",
            {"valid": True, "normalized": result.normalized_value, "next_field": next_state},
        )
    else:
        logger.info(
            "PHONE_VALIDATION %s",
            {"valid": False, "reason": result.reason, "next_field": next_state},
        )
    state.trace = dict(state.trace or {})
    state.trace["phone_validation"] = payload
    state.trace["validation_reason"] = result.reason


def _lead_from_state(state: ConversationState) -> Lead:
    state.city_country = state.city_country or CITY_COUNTRY_INDIA
    return Lead(
        name=state.user_name,
        phone=state.phone,
        city=state.city,
        country=state.phone_country or state.country,
        product=state.product,
        conversation_id=state.conversation_id,
    )


def _mark_lead_created(
    state: ConversationState,
    lead_id: str,
    *,
    tool_action: str = "create_lead",
) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_called"] = tool_action
    state.trace["should_create_lead"] = True
    state.trace["lead_id"] = lead_id
    logger.info(
        "%s %s",
        tool_action.upper(),
        {"called": True, "lead_id": lead_id, "city": state.city},
    )


def _add_capability(state: ConversationState, capability: str) -> None:
    state.trace = dict(state.trace or {})
    caps = list(state.trace.get("capabilities") or [])
    if capability not in caps:
        caps.append(capability)
    state.trace["capabilities"] = [item for item in caps if item != "CONVERSATION_ONLY"]


def _record_tool_ms(state: ConversationState, started: float) -> None:
    state.trace = dict(state.trace or {})
    state.trace["tool_ms"] = round((time.perf_counter() - started) * 1000, 3)
