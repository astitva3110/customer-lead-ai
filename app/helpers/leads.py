from __future__ import annotations

from app.domain.entities import Lead


def lead_summary_payload(lead: Lead) -> dict:
    return {
        "lead_id": lead.lead_id,
        "name": lead.name,
        "phone": lead.phone,
        "city": lead.city,
        "country": lead.country,
        "product": lead.product,
        "conversation_id": lead.conversation_id,
        "status": lead.status.value,
        "created_at": lead.created_at.isoformat() if lead.created_at else None,
    }
