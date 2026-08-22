from __future__ import annotations

import logging

import jwt

from app.config import settings
from app.domain.entities import User, UserRole
from app.helpers.jwt_tokens import create_access_token
from app.helpers.passwords import hash_password, verify_password
from app.interfaces.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, users: UserRepository) -> None:
        self._users = users

    def login(self, *, email: str, password: str) -> tuple[str, User]:
        user = self._users.get_by_email(email)
        if user is None or not verify_password(password, user.password_hash):
            raise ValueError("Invalid credentials")
        if not user.is_active:
            raise ValueError("User account is disabled")
        token = create_access_token(user_id=user.user_id, role=user.role)
        return token, user

    def get_user_from_token(self, token: str) -> User:
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
            )
        except jwt.PyJWTError as exc:
            raise ValueError("Invalid or expired token") from exc

        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Invalid token payload")

        user = self._users.get_by_id(str(user_id))
        if user is None:
            raise ValueError("User not found")
        if not user.is_active:
            raise ValueError("User account is disabled")
        return user


def bootstrap_initial_super_admin(users: UserRepository) -> None:
    email = settings.initial_super_admin_email.strip().lower()
    password = settings.initial_super_admin_password
    if not email or not password:
        return
    if users.get_by_email(email) is not None:
        return
    users.create_user(
        User(
            email=email,
            password_hash=hash_password(password),
            role=UserRole.SUPER_ADMIN,
            is_active=True,
        )
    )
    logger.info("Initial super admin account created for %s", email)
