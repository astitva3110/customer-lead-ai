from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ChatMode(StrEnum):
    KNOWLEDGE = "KNOWLEDGE"
    LEAD = "LEAD"
    SUPPORT = "SUPPORT"
    REJECTED = "REJECTED"


class ConversationGoal(StrEnum):
    NONE = "NONE"
    KNOWLEDGE = "KNOWLEDGE"
    LEAD = "LEAD"
    SALES = "LEAD"
    SUPPORT = "SUPPORT"


class TurnIntent(StrEnum):
    KNOWLEDGE = "KNOWLEDGE"
    LEAD_INTENT = "LEAD_INTENT"
    SALES = "LEAD_INTENT"
    SUPPORT_INTENT = "SUPPORT_INTENT"
    PROVIDE_INFORMATION = "PROVIDE_INFORMATION"
    CONTEXT_UPDATE = "CONTEXT_UPDATE"
    CONFIRMATION = "CONFIRMATION"
    ACTION = "ACTION"
    MIXED = "MIXED"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


class LeadWorkflow(StrEnum):
    NONE = "NONE"
    DISCUSSING_PRODUCT = "DISCUSSING_PRODUCT"
    COLLECTING_PHONE = "COLLECTING_PHONE"
    COLLECTING_NAME = "COLLECTING_NAME"
    COLLECTING_CITY = "COLLECTING_CITY"
    READY_TO_CREATE = "READY_TO_CREATE"
    CREATED = "CREATED"


class SupportWorkflow(StrEnum):
    NONE = "NONE"
    UNDERSTANDING_ISSUE = "UNDERSTANDING_ISSUE"
    COLLECTING_NAME = "COLLECTING_NAME"
    COLLECTING_PRODUCT = "COLLECTING_PRODUCT"
    COLLECTING_PHONE = "COLLECTING_PHONE"
    COLLECTING_ISSUE = "COLLECTING_ISSUE"
    READY_TO_CREATE = "READY_TO_CREATE"
    CREATED = "CREATED"


class LeadStatus(StrEnum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    CREATED = "CREATED"
    FAILED = "FAILED"


class TicketStatus(StrEnum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    CREATED = "CREATED"
    FAILED = "FAILED"


class LeadStage(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class ConversationState:
    conversation_id: str = ""
    user_message: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    intent: str = ""
    mode: str = ""
    return_mode: str = ""
    awaiting_field: str = ""
    conversation_goal: str = ConversationGoal.NONE
    current_turn_intent: str = TurnIntent.UNKNOWN
    lead_workflow: str = LeadWorkflow.NONE
    support_workflow: str = SupportWorkflow.NONE
    lead_intent: bool = False
    support_intent: bool = False
    sales_interest: bool = False
    lead_stage: str = LeadStage.NOT_STARTED
    lead_collection_active: bool = False
    explicit_action: str = ""
    user_context: dict[str, Any] = field(default_factory=dict)

    product: str = ""
    product_id: str = ""

    user_name: str = ""
    phone: str = ""
    country: str = ""
    phone_country: str = ""
    city: str = ""
    city_country: str = "IN"
    session_country: str = ""
    pending_phone: str = ""

    support_issue: str = ""

    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)

    lead_status: str = LeadStatus.IDLE
    ticket_status: str = TicketStatus.IDLE

    response: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    query_rewritten: str = ""
    guardrail_rejected: bool = False
    error: str = ""
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_message": self.user_message,
            "conversation_history": list(self.conversation_history),
            "intent": self.intent,
            "mode": self.mode,
            "return_mode": self.return_mode,
            "awaiting_field": self.awaiting_field,
            "conversation_goal": self.conversation_goal,
            "current_turn_intent": self.current_turn_intent,
            "lead_workflow": self.lead_workflow,
            "support_workflow": self.support_workflow,
            "lead_intent": self.lead_intent,
            "support_intent": self.support_intent,
            "sales_interest": self.sales_interest,
            "lead_stage": self.lead_stage,
            "lead_collection_active": self.lead_collection_active,
            "explicit_action": self.explicit_action,
            "user_context": dict(self.user_context),
            "product": self.product,
            "product_id": self.product_id,
            "user_name": self.user_name,
            "phone": self.phone,
            "country": self.country,
            "phone_country": self.phone_country,
            "city": self.city,
            "city_country": self.city_country,
            "session_country": self.session_country,
            "pending_phone": self.pending_phone,
            "support_issue": self.support_issue,
            "retrieved_context": list(self.retrieved_context),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieval_metadata": dict(self.retrieval_metadata),
            "lead_status": self.lead_status,
            "ticket_status": self.ticket_status,
            "response": self.response,
            "sources": list(self.sources),
            "query_rewritten": self.query_rewritten,
            "guardrail_rejected": self.guardrail_rejected,
            "error": self.error,
            "trace": dict(self.trace),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConversationState:
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)
