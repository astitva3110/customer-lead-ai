from __future__ import annotations

from app.domain.entities import UserRole

_ROLE_LEVEL = {
    UserRole.USER: 0,
    UserRole.ADMIN: 1,
    UserRole.SUPER_ADMIN: 2,
}


def role_level(role: UserRole) -> int:
    return _ROLE_LEVEL[role]


def role_at_least(user_role: UserRole, minimum_role: UserRole) -> bool:
    return role_level(user_role) >= role_level(minimum_role)


def role_in_allowed(user_role: UserRole, allowed_roles: tuple[UserRole, ...]) -> bool:
    if user_role in allowed_roles:
        return True
    minimum = min(role_level(role) for role in allowed_roles)
    return role_level(user_role) >= minimum
