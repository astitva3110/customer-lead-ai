import json

from app.services.conversation.models import ConversationGoal, TicketStatus
from tests.unit.conversation.fakes import RecordingLLM, make_orchestrator


def test_price_query_points_to_website_and_offers_lead() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("price-1", "What is the price of Bluup?")
    assert knowledge.queries == []
    assert "earkart.com" in result.response.lower()
    assert "our team" in result.response.lower()
    assert "great choice" not in result.response.lower()
    assert "create a lead" not in result.response.lower()
    assert result.trace.get("price_lead_offer") is True
    assert result.trace.get("pending_choice") == "lead_offer"


def test_decline_lead_offer_does_not_repeat_buy_pitch() -> None:
    from app.helpers.quick_replies import lead_choice_prompt

    pitch = json.dumps({"answer": lead_choice_prompt("TINY")})
    orchestrator, _, _, lead, *_ = make_orchestrator(llm=RecordingLLM(output=pitch))
    cid = "decline-lead-offer"
    first = orchestrator.handle(cid, "I want to buy TINY.")
    assert first.trace.get("pending_choice") == "lead_offer"
    second = orchestrator.handle(cid, "No")
    assert "no worries" in second.response.lower()
    assert "create a lead" not in second.response.lower()
    assert lead.leads == []
    assert not second.lead_collection_active


def test_yes_after_price_offer_starts_lead_collection() -> None:
    orchestrator, _, _, lead_tool, *_ = make_orchestrator()
    cid = "price-2"
    first = orchestrator.handle(cid, "How much does TINY cost?")
    assert "our team" in first.response.lower()
    second = orchestrator.handle(cid, "yes")
    assert second.lead_collection_active or second.explicit_action == "create_lead"
    assert second.awaiting_field in {"name", "city", "phone", "phone_country"}
    assert lead_tool.leads == []


def test_known_name_skipped_when_starting_lead() -> None:
    orchestrator, _, _, lead_tool, *_ = make_orchestrator()
    cid = "known-name-lead"
    intro = orchestrator.handle(cid, "I am Rahul.")
    assert intro.user_name == "Rahul"
    callback = orchestrator.handle(cid, "Please call me.")
    assert callback.user_name == "Rahul"
    assert callback.awaiting_field == "phone"
    assert "name should" not in callback.response.lower()
    assert lead_tool.leads == []


def test_whatsapp_support_ticket_skips_known_name_and_phone() -> None:
    orchestrator, _, _, _, ticket_tool, _ = make_orchestrator()
    first = orchestrator.handle(
        None,
        "My TINY is not working.",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
        user_name="Rahul",
    )
    assert first.user_name == "Rahul"
    assert first.phone == "+919876543210"
    assert first.conversation_goal == ConversationGoal.SUPPORT

    ticket = orchestrator.handle(
        first.conversation_id,
        "Please create a support ticket.",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
        user_name="Rahul",
    )
    assert ticket.support_collection_active or ticket.explicit_action == "create_ticket"
    assert ticket.awaiting_field in {"product", "issue", ""}
    assert "name should" not in ticket.response.lower()
    assert "number" not in ticket.response.lower()
    if ticket.ticket_status == TicketStatus.CREATED:
        assert ticket_tool.tickets[0].name == "Rahul"
        assert ticket_tool.tickets[0].phone == "+919876543210"
