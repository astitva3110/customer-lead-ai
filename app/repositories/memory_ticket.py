from __future__ import annotations

from app.domain.entities import SupportTicket


class InMemoryTicketAdapter:
    def __init__(self) -> None:
        self.tickets: list[SupportTicket] = []

    def create_support_ticket(self, ticket: SupportTicket) -> str:
        self.tickets.append(ticket)
        return ticket.ticket_id
