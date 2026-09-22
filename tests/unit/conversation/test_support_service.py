from dataclasses import dataclass, field

from app.helpers.conversation_reply import ticket_created_reply
from app.helpers.crm_lead import CRM_PROBLEM_CUSTOMER_SERVICE
from app.repositories.memory_ticket import InMemoryTicketAdapter
from app.services.conversation.models import ConversationState, TicketStatus
from app.services.conversation.support_service import SupportService


@dataclass
class FakeCrm:
    calls: list[dict[str, str]] = field(default_factory=list)
    fail: bool = False

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
        if self.fail:
            raise RuntimeError("crm unavailable")
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


def test_missing_phone_prompts_after_name() -> None:
    service = SupportService(InMemoryTicketAdapter())
    state = service.handle(
        ConversationState(user_message="My hearing aid isn't working.", user_name="Ada")
    )
    assert state.ticket_status == TicketStatus.COLLECTING
    assert state.support_collection_active
    assert state.awaiting_field == "phone"
    assert state.support_issue


def test_missing_product_prompts_after_phone() -> None:
    service = SupportService(InMemoryTicketAdapter())
    state = ConversationState(
        user_message="Radius M16",
        user_name="Ada",
        phone="+919876543210",
        support_issue="not working",
        awaiting_field="product",
        ticket_status=TicketStatus.COLLECTING,
        support_collection_active=True,
    )
    state = service.handle(state)
    assert state.product == "Radius M16"
    assert state.awaiting_field == ""


def test_ticket_creation() -> None:
    adapter = InMemoryTicketAdapter()
    state = ConversationState(
        user_message="+91 9876543210",
        user_name="Ada",
        product="Radius M16",
        support_issue="My hearing aid isn't working.",
        awaiting_field="phone",
        ticket_status=TicketStatus.COLLECTING,
        support_collection_active=True,
        conversation_id="c1",
    )
    state = SupportService(adapter).handle(state)
    assert state.ticket_status == TicketStatus.CREATED
    assert state.response == ticket_created_reply("Ada")
    assert adapter.tickets[0].product == "Radius M16"
    assert state.trace["ticket_id"] == adapter.tickets[0].ticket_id


def test_ticket_creation_syncs_crm_with_customer_service_problem() -> None:
    adapter = InMemoryTicketAdapter()
    crm = FakeCrm()
    state = ConversationState(
        user_message="+91 9876543210",
        user_name="Ada",
        product="Radius M16",
        support_issue="My hearing aid isn't working.",
        awaiting_field="phone",
        ticket_status=TicketStatus.COLLECTING,
        support_collection_active=True,
        conversation_id="support-crm",
        channel="whatsapp",
    )
    state = SupportService(adapter, crm=crm).handle(state)
    assert state.ticket_status == TicketStatus.CREATED
    assert state.trace["crm_synced"] is True
    assert crm.calls
    assert crm.calls[0]["names"] == "Ada"
    assert crm.calls[0]["phone"] == "9876543210"
    assert crm.calls[0]["source"] == "whatsapp"
    assert crm.calls[0]["problem"] == CRM_PROBLEM_CUSTOMER_SERVICE
    assert crm.calls[0]["city"] == "null"


def test_crm_failure_still_creates_ticket_locally() -> None:
    adapter = InMemoryTicketAdapter()
    crm = FakeCrm(fail=True)
    state = ConversationState(
        user_message="+91 9876543210",
        user_name="Ada",
        product="Radius M16",
        support_issue="not working",
        awaiting_field="phone",
        ticket_status=TicketStatus.COLLECTING,
        support_collection_active=True,
        conversation_id="support-crm-fail",
    )
    state = SupportService(adapter, crm=crm).handle(state)
    assert state.ticket_status == TicketStatus.CREATED
    assert adapter.tickets
    assert state.trace["crm_synced"] is False
