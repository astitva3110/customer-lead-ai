from app.services.conversation.lead_service import LeadService
from app.helpers.conversation_reply import lead_created_reply
from app.services.conversation.models import ConversationState, LeadStatus
from app.repositories.memory_lead import InMemoryLeadAdapter


def test_new_lead_prompts_for_name_first() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = service.handle(ConversationState(user_message="I want to buy Radius M16."))
    assert state.awaiting_field == "name"
    assert state.lead_status == LeadStatus.COLLECTING
    assert state.product == "Radius M16"
    assert "name" in state.response.lower()
    assert "number" not in state.response.lower()


def test_invalid_phone_keeps_asking() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = ConversationState(
        user_message="not-a-phone",
        product="Radius M16",
        awaiting_field="phone",
        lead_status=LeadStatus.COLLECTING,
    )
    state = service.handle(state)
    assert not state.phone
    assert state.awaiting_field == "phone"
    assert state.lead_status != LeadStatus.CREATED


def test_valid_phone_then_asks_for_name() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = ConversationState(
        user_message="+91 9876543210",
        product="Radius M16",
        awaiting_field="phone",
        lead_status=LeadStatus.COLLECTING,
    )
    state = service.handle(state)
    assert state.phone == "+919876543210"
    assert state.country == "IN"
    assert state.awaiting_field == "name"


def test_missing_name_prompts_for_name() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = ConversationState(
        user_message="Ada",
        product="Radius M16",
        phone="+919876543210",
        country="IN",
        awaiting_field="name",
        lead_status=LeadStatus.COLLECTING,
    )
    state = service.handle(state)
    assert state.user_name == "Ada"
    assert state.awaiting_field == "city"


def test_lead_creation() -> None:
    adapter = InMemoryLeadAdapter()
    state = ConversationState(
        user_message="Noida",
        product="Radius M16",
        phone="+919876543210",
        country="IN",
        user_name="Ada",
        awaiting_field="city",
        lead_status=LeadStatus.COLLECTING,
        conversation_id="c1",
    )
    state = LeadService(adapter).handle(state)
    assert state.city == "Noida"
    assert state.lead_status == LeadStatus.CREATED
    assert state.response == lead_created_reply("Ada")
    assert adapter.leads[0].product == "Radius M16"
    assert state.trace["lead_id"] == adapter.leads[0].lead_id
