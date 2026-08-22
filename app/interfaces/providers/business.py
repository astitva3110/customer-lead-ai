from __future__ import annotations

from typing import Protocol

from app.domain.entities import Lead, SupportTicket


class LeadTool(Protocol):
    def create_lead(self, lead: Lead) -> str: ...


class TicketTool(Protocol):
    def create_support_ticket(self, ticket: SupportTicket) -> str: ...
