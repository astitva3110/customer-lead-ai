from app.domain.entities import Lead, SupportTicket
from app.helpers.crm_lead import (
    CRM_CITY_PLACEHOLDER,
    CRM_PROBLEM_CUSTOMER_SERVICE,
    CRM_PROBLEM_SALES,
    crm_city_value,
    crm_problem_from_lead,
    crm_problem_from_ticket,
)


def test_crm_city_value_uses_null_placeholder_when_missing() -> None:
    assert crm_city_value("") == CRM_CITY_PLACEHOLDER
    assert crm_city_value(None) == CRM_CITY_PLACEHOLDER
    assert crm_city_value("Noida") == "Noida"


def test_crm_problem_from_lead_is_sales() -> None:
    lead = Lead(
        name="Ada",
        phone="+919876543210",
        product="TINY",
        city="Noida",
        country="IN",
        conversation_id="c1",
    )
    assert crm_problem_from_lead(lead) == CRM_PROBLEM_SALES


def test_crm_problem_from_ticket_is_customer_service() -> None:
    ticket = SupportTicket(
        name="Ada",
        phone="+919876543210",
        product="TINY",
        issue="not working",
        conversation_id="c1",
    )
    assert crm_problem_from_ticket(ticket) == CRM_PROBLEM_CUSTOMER_SERVICE
