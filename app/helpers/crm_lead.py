from __future__ import annotations

import logging

from app.domain.entities import Lead, SupportTicket
from app.helpers.phone import digits_only
from app.interfaces.providers.crm import CrmLeadPort
from app.services.conversation.models import ConversationState

logger = logging.getLogger(__name__)

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


def sync_to_crm(
    crm: CrmLeadPort | None,
    *,
    state: ConversationState,
    names: str,
    phone: str,
    problem: str,
    city: str | None = None,
) -> bool:
    """Push a lead or support ticket to CRM. Returns True when the CRM call succeeds."""
    if crm is None:
        return False
    phone_digits = crm_phone_digits(phone)
    if not phone_digits:
        return False
    try:
        crm.create_lead(
            names=(names or "").strip(),
            phone=phone_digits,
            source=crm_source_for_state(state),
            problem=problem,
            city=crm_city_value(city),
        )
    except Exception:
        logger.exception("crm sync failed problem=%s phone=%s", problem, phone_digits[-4:])
        return False
    return True
