from __future__ import annotations

from app.domain.entities import User


class InMemoryUserRepository:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def get_by_id(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def get_by_email(self, email: str) -> User | None:
        normalized = email.lower()
        for user in self._users.values():
            if user.email == normalized:
                return user
        return None

    def list_users(self) -> list[User]:
        return list(self._users.values())

    def create_user(self, user: User) -> User:
        self._users[user.user_id] = user
        return user

    def update_user(self, user: User) -> User:
        if user.user_id not in self._users:
            raise ValueError(f"User not found: {user.user_id}")
        self._users[user.user_id] = user
        return user
