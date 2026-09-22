from __future__ import annotations

from app.domain.entities import Lead, SupportTicket
from app.helpers.phone import digits_only
from app.services.conversation.models import ConversationState

CRM_PROBLEM_SALES = "Sales"
CRM_PROBLEM_CUSTOMER_SERVICE = "Customer service"
CRM_CITY_PLACEHOLDER = "null"


def crm_city_value(city: str | None) -> str:
    """CRM requires city; send literal 'null' when we do not have one yet."""
    text = (city or "").strip()
    return text if text else CRM_CITY_PLACEHOLDER


def crm_source_for_state(state: ConversationState) -> str:
    channel = (state.channel or "").strip().lower()
    if channel == "whatsapp":
        return "whatsapp"
    return "web"


def crm_phone_digits(phone: str) -> str:
    text = (phone or "").strip()
    if text.startswith("+91") and len(text) > 3:
        return digits_only(text[3:]) or digits_only(text)
    return digits_only(text) or text


def crm_problem_from_lead(_lead: Lead) -> str:
    return CRM_PROBLEM_SALES


def crm_problem_from_ticket(_ticket: SupportTicket) -> str:
    return CRM_PROBLEM_CUSTOMER_SERVICE


def crm_names_from_lead(lead: Lead) -> str:
    return (lead.name or "").strip()


def crm_names_from_ticket(ticket: SupportTicket) -> str:
    return (ticket.name or "").strip()
