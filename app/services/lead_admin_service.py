from __future__ import annotations

from app.domain.entities import Lead, LeadRecordStatus
from app.repositories.lead import PostgresLeadRepository


class LeadAdminService:
    def __init__(self, leads: PostgresLeadRepository) -> None:
        self._leads = leads

    def list_leads(self, *, limit: int, offset: int) -> list[Lead]:
        return self._leads.list_leads(limit=limit, offset=offset)

    def get_lead(self, lead_id: str) -> Lead | None:
        return self._leads.get_lead(lead_id)

    def update_status(self, lead_id: str, status: LeadRecordStatus) -> Lead | None:
        return self._leads.update_lead_status(lead_id, status)
