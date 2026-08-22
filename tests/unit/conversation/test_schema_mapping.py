from app.domain.entities import Lead, SupportTicket
from app.db.models import LeadRow, SupportTicketRow


def test_lead_row_maps_to_and_from_entity() -> None:
    lead = Lead(
        lead_id="lead-1",
        name="Ada",
        phone="+919876543210",
        city="Noida",
        country="IN",
        product="TINY",
        conversation_id="c1",
    )
    row = LeadRow.from_entity(lead)
    assert row.to_entity() == lead


def test_ticket_row_maps_to_and_from_entity() -> None:
    ticket = SupportTicket(
        ticket_id="ticket-1",
        name="Ada",
        phone="+919876543210",
        product="TINY",
        issue="Not charging",
        conversation_id="c1",
    )
    row = SupportTicketRow.from_entity(ticket)
    assert row.to_entity() == ticket
