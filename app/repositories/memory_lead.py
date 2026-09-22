from __future__ import annotations

from app.domain.entities import Lead


class InMemoryLeadAdapter:
    def __init__(self) -> None:
        self.leads: list[Lead] = []

    def create_lead(self, lead: Lead) -> str:
        self.leads.append(lead)
        return lead.lead_id

    def update_lead(self, lead: Lead) -> Lead | None:
        for index, existing in enumerate(self.leads):
            if existing.lead_id != lead.lead_id:
                continue
            self.leads[index] = lead
            return lead
        return None
