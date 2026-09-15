from __future__ import annotations

from app.domain.entities import User, UserRole
from app.helpers.passwords import hash_password
from app.interfaces.repositories.user_repository import UserRepository


class UserAdminService:
    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def list_users(self) -> list[User]:
        return self._users.list_users()

    def create_user(
        self,
        *,
        email: str,
        password: str,
        role: UserRole,
        actor: User,
    ) -> User:
        if role == UserRole.SUPER_ADMIN and actor.role != UserRole.SUPER_ADMIN:
            raise PermissionError("Only super admins can create super admins")
        if self._users.get_by_email(email) is not None:
            raise ValueError("Email already registered")
        return self._users.create_user(
            User(
                email=email.lower(),
                password_hash=hash_password(password),
                role=role,
                is_active=True,
            )
        )

    def update_user(
        self,
        *,
        user_id: str,
        email: str | None,
        password: str | None,
        actor: User,
    ) -> User:
        user = self._users.get_by_id(user_id)
        if user is None:
            raise ValueError("User not found")
        if email is not None:
            user.email = email.lower()
        if password is not None:
            user.password_hash = hash_password(password)
        return self._users.update_user(user)

    def set_active(self, *, user_id: str, is_active: bool, actor: User) -> User:
        user = self._users.get_by_id(user_id)
        if user is None:
            raise ValueError("User not found")
        if user.user_id == actor.user_id and not is_active:
            raise ValueError("Cannot deactivate your own account")
        user.is_active = is_active
        return self._users.update_user(user)

    def set_role(self, *, user_id: str, role: UserRole, actor: User) -> User:
        user = self._users.get_by_id(user_id)
        if user is None:
            raise ValueError("User not found")
        if role == UserRole.SUPER_ADMIN and actor.role != UserRole.SUPER_ADMIN:
            raise PermissionError("Only super admins can assign super admin role")
        if user.role == UserRole.SUPER_ADMIN and actor.role != UserRole.SUPER_ADMIN:
            raise PermissionError("Only super admins can manage super admins")
        user.role = role
        return self._users.update_user(user)
