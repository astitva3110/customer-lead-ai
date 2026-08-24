from __future__ import annotations

from app.domain.entities import ChatConversation, Lead, RecordStatus, SupportTicket
from app.repositories.chat_trace import PostgresChatTraceRepository
from app.repositories.lead import PostgresLeadRepository
from app.repositories.ticket import PostgresTicketRepository


class LeadAdminService:
    def __init__(
        self,
        leads: PostgresLeadRepository,
        traces: PostgresChatTraceRepository,
    ) -> None:
        self._leads = leads
        self._traces = traces

    def list_leads(
        self,
        *,
        limit: int,
        offset: int,
        status: RecordStatus | None = None,
    ) -> list[Lead]:
        return self._leads.list_leads(limit=limit, offset=offset, status=status)

    def get_lead(self, lead_id: str) -> Lead | None:
        return self._leads.get_lead(lead_id)

    def get_lead_detail(self, lead_id: str) -> tuple[Lead, ChatConversation | None] | None:
        lead = self._leads.get_lead(lead_id)
        if lead is None:
            return None
        conversation = self._traces.get_conversation(lead.conversation_id)
        return lead, conversation

    def update_status(self, lead_id: str, status: RecordStatus) -> Lead | None:
        return self._leads.update_lead_status(lead_id, status)


class SupportAdminService:
    def __init__(
        self,
        tickets: PostgresTicketRepository,
        traces: PostgresChatTraceRepository,
    ) -> None:
        self._tickets = tickets
        self._traces = traces

    def list_tickets(
        self,
        *,
        limit: int,
        offset: int,
        status: RecordStatus | None = None,
    ) -> list[SupportTicket]:
        return self._tickets.list_tickets(limit=limit, offset=offset, status=status)

    def get_ticket_detail(self, ticket_id: str) -> tuple[SupportTicket, ChatConversation | None] | None:
        ticket = self._tickets.get_ticket(ticket_id)
        if ticket is None:
            return None
        conversation = self._traces.get_conversation(ticket.conversation_id)
        return ticket, conversation

    def update_status(self, ticket_id: str, status: RecordStatus) -> SupportTicket | None:
        return self._tickets.update_ticket_status(ticket_id, status)
