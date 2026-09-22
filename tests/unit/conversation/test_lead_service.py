from __future__ import annotations

from dataclasses import dataclass, field

from app.helpers.conversation_reply import lead_created_reply
from app.helpers.crm_lead import CRM_CITY_PLACEHOLDER, CRM_PROBLEM_SALES
from app.repositories.memory_lead import InMemoryLeadAdapter
from app.services.conversation.lead_service import LeadService
from app.services.conversation.models import ConversationState, LeadStatus


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


def test_city_with_pincode_is_saved_and_creates_lead() -> None:
    adapter = InMemoryLeadAdapter()
    state = ConversationState(
        user_message="Gurugram-122006",
        product="TINY",
        phone="+919310707781",
        country="IN",
        user_name="MC Sharma",
        awaiting_field="city",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="city-gurugram-pin",
    )
    state = LeadService(adapter).handle(state)
    assert state.city == "Gurugram-122006"
    assert state.lead_status == LeadStatus.CREATED
    assert adapter.leads
    assert adapter.leads[0].city == "Gurugram-122006"


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
    assert state.trace["lead_persisted"] is True
    assert state.trace["tool_called"] == "update_lead"
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
    assert state.trace["tool_called"] == "update_lead"


def test_phone_without_city_persists_and_asks_city() -> None:
    adapter = InMemoryLeadAdapter()
    crm = FakeCrm()
    state = ConversationState(
        user_message="+91 9876543210",
        product="Radius M16",
        user_name="Ada",
        awaiting_field="phone",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="persist-phone",
    )
    state = LeadService(adapter, crm=crm).handle(state)
    assert state.phone == "+919876543210"
    assert state.lead_status == LeadStatus.COLLECTING
    assert state.awaiting_field == "city"
    assert state.response != lead_created_reply("Ada")
    assert "city" in state.response.lower()
    assert adapter.leads
    assert adapter.leads[0].city == ""
    assert state.trace["lead_id"] == adapter.leads[0].lead_id
    assert state.trace["crm_synced"] is True
    assert crm.calls
    assert crm.calls[0]["phone"] == "9876543210"
    assert crm.calls[0]["source"] == "web"
    assert crm.calls[0]["problem"] == CRM_PROBLEM_SALES
    assert crm.calls[0].get("city", CRM_CITY_PLACEHOLDER) == CRM_CITY_PLACEHOLDER


def test_city_after_persist_updates_lead_and_shows_success() -> None:
    adapter = InMemoryLeadAdapter()
    crm = FakeCrm()
    service = LeadService(adapter, crm=crm)
    phone_state = ConversationState(
        user_message="+91 9876543210",
        user_name="Ada",
        awaiting_field="phone",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="complete-flow",
    )
    phone_state = service.handle(phone_state)
    lead_id = phone_state.trace["lead_id"]
    city_state = ConversationState(
        user_message="Noida",
        product="Radius M16",
        phone=phone_state.phone,
        country="IN",
        user_name="Ada",
        awaiting_field="city",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="complete-flow",
        trace=dict(phone_state.trace),
    )
    city_state = service.handle(city_state)
    assert city_state.lead_status == LeadStatus.CREATED
    assert city_state.response == lead_created_reply("Ada")
    assert len(adapter.leads) == 1
    assert adapter.leads[0].lead_id == lead_id
    assert adapter.leads[0].city == "Noida"
    assert len(crm.calls) == 1


def test_whatsapp_prefilled_phone_persists_after_name() -> None:
    adapter = InMemoryLeadAdapter()
    crm = FakeCrm()
    state = ConversationState(
        user_message="Ravi",
        channel="whatsapp",
        phone="+919876543210",
        country="IN",
        awaiting_field="name",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="wa-name",
    )
    state = LeadService(adapter, crm=crm).handle(state)
    assert state.user_name == "Ravi"
    assert state.awaiting_field == "city"
    assert state.lead_status == LeadStatus.COLLECTING
    assert state.trace["lead_id"]
    assert crm.calls[0]["source"] == "whatsapp"


def test_crm_failure_still_persists_locally_and_asks_city() -> None:
    adapter = InMemoryLeadAdapter()
    crm = FakeCrm(fail=True)
    state = ConversationState(
        user_message="+91 9876543210",
        user_name="Ada",
        awaiting_field="phone",
        lead_collection_active=True,
        lead_status=LeadStatus.COLLECTING,
        conversation_id="crm-fail",
    )
    state = LeadService(adapter, crm=crm).handle(state)
    assert adapter.leads
    assert state.trace["lead_id"] == adapter.leads[0].lead_id
    assert state.trace["crm_synced"] is False
    assert state.awaiting_field == "city"
    assert state.lead_status == LeadStatus.COLLECTING
