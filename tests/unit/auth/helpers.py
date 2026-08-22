from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

from app.dependencies import get_auth_service, get_current_user, get_user_admin_service, get_user_repository
from app.domain.entities import User, UserRole
from app.helpers.jwt_tokens import create_access_token
from app.helpers.passwords import hash_password
from app.main import app
from app.repositories.memory_user import InMemoryUserRepository
from app.services.auth_service import AuthService
from app.services.user_admin_service import UserAdminService


def make_user(
    *,
    email: str | None = None,
    password: str = "password123",
    role: UserRole = UserRole.USER,
    is_active: bool = True,
) -> tuple[User, str]:
    user = User(
        user_id=str(uuid4()),
        email=(email or f"{role.value}-{uuid4().hex[:8]}@example.com").lower(),
        password_hash=hash_password(password),
        role=role,
        is_active=is_active,
    )
    return user, password


def auth_header(user: User) -> dict[str, str]:
    token = create_access_token(user_id=user.user_id, role=user.role)
    return {"Authorization": f"Bearer {token}"}


def clear_auth_caches() -> None:
    get_user_repository.cache_clear()
    get_auth_service.cache_clear()
    get_user_admin_service.cache_clear()


@contextmanager
def user_repo(users: list[User] | None = None):
    repo = InMemoryUserRepository()
    for user in users or []:
        repo.create_user(user)
    clear_auth_caches()
    app.dependency_overrides[get_user_repository] = lambda: repo
    app.dependency_overrides[get_auth_service] = lambda: AuthService(repo)
    app.dependency_overrides[get_user_admin_service] = lambda: UserAdminService(repo)
    try:
        yield repo
    finally:
        app.dependency_overrides.pop(get_user_repository, None)
        app.dependency_overrides.pop(get_auth_service, None)
        app.dependency_overrides.pop(get_user_admin_service, None)
        clear_auth_caches()


@contextmanager
def as_user(user: User):
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)
