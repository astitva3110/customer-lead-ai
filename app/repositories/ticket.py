from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.domain.entities import RecordStatus, SupportTicket
from app.db.engine import get_session_factory
from app.db.models import SupportTicketRow, _utcnow


class PostgresTicketRepository:
    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def create_support_ticket(self, ticket: SupportTicket) -> str:
        with self._session_factory() as session:
            session.add(SupportTicketRow.from_entity(ticket))
            session.commit()
        return ticket.ticket_id

    def list_tickets(
        self,
        *,
        limit: int,
        offset: int,
        status: RecordStatus | None = None,
    ) -> list[SupportTicket]:
        with self._session_factory() as session:
            stmt = (
                select(SupportTicketRow)
                .order_by(SupportTicketRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if status is not None:
                stmt = stmt.where(SupportTicketRow.status == status.value)
            rows = session.execute(stmt).scalars().all()
            return [row.to_entity() for row in rows]

    def get_ticket(self, ticket_id: str) -> SupportTicket | None:
        with self._session_factory() as session:
            row = session.get(SupportTicketRow, ticket_id)
            return row.to_entity() if row else None

    def update_ticket_status(self, ticket_id: str, status: RecordStatus) -> SupportTicket | None:
        with self._session_factory() as session:
            row = session.get(SupportTicketRow, ticket_id)
            if row is None:
                return None
            row.status = status.value
            row.updated_at = _utcnow()
            session.commit()
            session.refresh(row)
            return row.to_entity()
