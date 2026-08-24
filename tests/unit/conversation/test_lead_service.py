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


def test_invalid_phone_length_explains_failure() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = ConversationState(
        user_message="12345",
        user_name="sudhanshu",
        awaiting_field="phone",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
    )
    state = service.handle(state)
    assert not state.phone
    assert state.awaiting_field == "phone"
    assert state.trace["validation_reason"] == "invalid_length"
    assert "10-digit" in state.response.lower()
    assert "which city" not in state.response.lower()
    assert "absolutely" not in state.response.lower()


def test_invalid_phone_prefix_does_not_advance() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = ConversationState(
        user_message="1234567890",
        user_name="Sudhanshu",
        awaiting_field="phone",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
    )
    state = service.handle(state)
    assert not state.phone
    assert state.awaiting_field == "phone"
    assert state.trace["validation_reason"] == "invalid_prefix"
    assert "indian mobile" in state.response.lower()
    assert "which city" not in state.response.lower()


def test_name_then_phone_uses_natural_thanks() -> None:
    service = LeadService(InMemoryLeadAdapter())
    state = ConversationState(
        user_message="sudhanshu",
        awaiting_field="name",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        explicit_action="create_lead",
    )
    state = service.handle(state)
    assert state.user_name == "Sudhanshu"
    assert state.awaiting_field == "phone"
    assert state.response == "Thanks, Sudhanshu. What's the best number to reach you on?"
    assert "absolutely" not in state.response.lower()


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
    assert state.phone_country == "IN"
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


def test_plain_city_is_saved_and_creates_lead() -> None:
    adapter = InMemoryLeadAdapter()
    state = ConversationState(
        user_message="Jaipur",
        product="TINY",
        phone="+919876543210",
        country="IN",
        user_name="Sudhanshu",
        awaiting_field="city",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="city-jaipur",
    )
    state = LeadService(adapter).converse(state)
    assert state.city == "Jaipur"
    assert state.trace["lead_missing"]["missing_fields"] == []
    assert state.trace["tool_called"] == "create_lead"
    assert adapter.leads
    assert adapter.leads[0].city == "Jaipur"


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
