from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from app.domain.entities import SupportTicket
from app.db.engine import get_session_factory
from app.db.models import SupportTicketRow


class PostgresTicketRepository:
    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def create_support_ticket(self, ticket: SupportTicket) -> str:
        with self._session_factory() as session:
            session.add(SupportTicketRow.from_entity(ticket))
            session.commit()
        return ticket.ticket_id
