from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.domain.entities import Lead, LeadRecordStatus
from app.db.engine import get_session_factory
from app.db.models import LeadRow


class PostgresLeadRepository:
    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def create_lead(self, lead: Lead) -> str:
        with self._session_factory() as session:
            session.add(LeadRow.from_entity(lead))
            session.commit()
        return lead.lead_id

    def list_leads(self, *, limit: int, offset: int) -> list[Lead]:
        with self._session_factory() as session:
            rows = (
                session.execute(
                    select(LeadRow).order_by(LeadRow.created_at.desc()).limit(limit).offset(offset)
                )
                .scalars()
                .all()
            )
            return [row.to_entity() for row in rows]

    def get_lead(self, lead_id: str) -> Lead | None:
        with self._session_factory() as session:
            row = session.get(LeadRow, lead_id)
            return row.to_entity() if row else None

    def update_lead_status(self, lead_id: str, status: LeadRecordStatus) -> Lead | None:
        with self._session_factory() as session:
            row = session.get(LeadRow, lead_id)
            if row is None:
                return None
            row.status = status.value
            session.commit()
            session.refresh(row)
            return row.to_entity()
