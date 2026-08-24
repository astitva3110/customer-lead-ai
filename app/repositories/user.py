from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.domain.entities import User
from app.db.engine import get_session_factory
from app.db.models import UserRow


class PostgresUserRepository:
    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def get_by_id(self, user_id: str) -> User | None:
        with self._session_factory() as session:
            row = session.get(UserRow, user_id)
            return row.to_entity() if row else None

    def get_by_email(self, email: str) -> User | None:
        with self._session_factory() as session:
            row = session.execute(
                select(UserRow).where(UserRow.email == email.lower())
            ).scalar_one_or_none()
            return row.to_entity() if row else None

    def list_users(self) -> list[User]:
        with self._session_factory() as session:
            rows = session.execute(select(UserRow).order_by(UserRow.created_at)).scalars().all()
            return [row.to_entity() for row in rows]

    def create_user(self, user: User) -> User:
        with self._session_factory() as session:
            row = UserRow.from_entity(user)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.to_entity()

    def update_user(self, user: User) -> User:
        with self._session_factory() as session:
            row = session.get(UserRow, user.user_id)
            if row is None:
                raise ValueError(f"User not found: {user.user_id}")
            row.email = user.email.lower()
            row.password_hash = user.password_hash
            row.role = user.role.value
            row.is_active = user.is_active
            session.commit()
            session.refresh(row)
            return row.to_entity()
