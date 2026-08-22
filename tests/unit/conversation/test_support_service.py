from app.services.conversation.models import ConversationState, TicketStatus
from app.services.conversation.support_service import SupportService
from app.helpers.conversation_reply import ticket_created_reply
from app.repositories.memory_ticket import InMemoryTicketAdapter


def test_missing_product_prompts_for_product() -> None:
    service = SupportService(InMemoryTicketAdapter())
    state = service.handle(
        ConversationState(user_message="My hearing aid isn't working.", user_name="Ada")
    )
    assert state.ticket_status == TicketStatus.COLLECTING
    assert state.awaiting_field == "product"
    assert state.support_issue


def test_missing_phone_prompts_for_phone() -> None:
    service = SupportService(InMemoryTicketAdapter())
    state = ConversationState(
        user_message="Radius M16",
        user_name="Ada",
        support_issue="not working",
        awaiting_field="product",
        ticket_status=TicketStatus.COLLECTING,
    )
    state = service.handle(state)
    assert state.product == "Radius M16"
    assert state.awaiting_field == "phone"


def test_ticket_creation() -> None:
    adapter = InMemoryTicketAdapter()
    state = ConversationState(
        user_message="+91 9876543210",
        user_name="Ada",
        product="Radius M16",
        support_issue="My hearing aid isn't working.",
        awaiting_field="phone",
        ticket_status=TicketStatus.COLLECTING,
        conversation_id="c1",
    )
    state = SupportService(adapter).handle(state)
    assert state.ticket_status == TicketStatus.CREATED
    assert state.response == ticket_created_reply("Ada")
    assert adapter.tickets[0].product == "Radius M16"
    assert state.trace["ticket_id"] == adapter.tickets[0].ticket_id
