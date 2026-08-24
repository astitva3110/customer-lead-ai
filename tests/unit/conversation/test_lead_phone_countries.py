from app.repositories.memory_lead import InMemoryLeadAdapter
from app.services.conversation.lead_service import LeadService
from app.services.conversation.models import ConversationState, LeadStatus


def _collecting(user_message: str, **kwargs) -> ConversationState:
    defaults = {
        "user_name": "Sudhanshu",
        "product": "TINY",
        "awaiting_field": "phone",
        "lead_collection_active": True,
        "lead_status": LeadStatus.COLLECTING,
        "conversation_id": "lead-phone",
    }
    defaults.update(kwargs)
    return ConversationState(user_message=user_message, **defaults)


def test_india_national_number_then_city_creates_lead() -> None:
    adapter = InMemoryLeadAdapter()
    service = LeadService(adapter)
    state = service.handle(_collecting("9876543210"))
    assert state.phone == "+919876543210"
    assert state.phone_country == "IN"
    assert state.country == "IN"
    assert state.awaiting_field == "city"
    assert state.response == "Thanks! Which city should I note for the team?"

    state.user_message = "Delhi"
    state = service.handle(state)
    assert state.city == "Delhi"
    assert state.lead_status == LeadStatus.CREATED
    assert adapter.leads[0].phone == "+919876543210"
    assert adapter.leads[0].city == "Delhi"


def test_invalid_length_does_not_advance_to_city() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = service.handle(_collecting("12345"))
    assert not state.phone
    assert state.awaiting_field == "phone"
    assert state.trace["phone_validation"]["valid"] is False
    assert state.trace["phone_validation"]["reason"] == "invalid_length"
    assert "10-digit" in state.response.lower()
    assert "which city" not in state.response.lower()


def test_invalid_prefix_does_not_advance_to_city() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = service.handle(_collecting("1234567890"))
    assert not state.phone
    assert state.awaiting_field == "phone"
    assert state.trace["phone_validation"]["reason"] == "invalid_prefix"
    assert "indian mobile" in state.response.lower()
