from dataclasses import dataclass, field

from app.domain.entities import Lead, SupportTicket
from app.helpers.crm_lead import (
    CRM_CITY_PLACEHOLDER,
    CRM_PROBLEM_CUSTOMER_SERVICE,
    CRM_PROBLEM_SALES,
    crm_city_value,
    crm_problem_from_lead,
    crm_problem_from_ticket,
    sync_to_crm,
)
from app.services.conversation.models import ConversationState


@dataclass
class _FakeCrm:
    calls: list[dict[str, str]] = field(default_factory=list)

    def create_lead(
        self,
        *,
        names: str,
        phone: str,
        source: str,
        email: str = "",
        problem: str = "",
        city: str = "null",
    ) -> None:
        self.calls.append(
            {
                "names": names,
                "phone": phone,
                "source": source,
                "email": email,
                "problem": problem,
                "city": city,
            }
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


def test_sync_to_crm_sends_null_city_when_missing() -> None:
    crm = _FakeCrm()
    state = ConversationState(conversation_id="c1", channel="whatsapp")
    synced = sync_to_crm(
        crm,
        state=state,
        names="Ada",
        phone="+919876543210",
        problem=CRM_PROBLEM_SALES,
        city=None,
    )
    assert synced is True
    assert crm.calls[0]["city"] == CRM_CITY_PLACEHOLDER


def test_crm_problem_from_ticket_is_customer_service() -> None:
    ticket = SupportTicket(
        name="Ada",
        phone="+919876543210",
        product="TINY",
        issue="not working",
        conversation_id="c1",
    )
    assert crm_problem_from_ticket(ticket) == CRM_PROBLEM_CUSTOMER_SERVICE
