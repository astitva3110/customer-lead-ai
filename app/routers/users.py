from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_current_user, get_user_admin_service, require_role
from app.domain.entities import User, UserRole
from app.schemas import (
    CreateUserRequest,
    UpdateUserRequest,
    UserListResponse,
    UserResponse,
    UserRoleUpdateRequest,
    UserStatusUpdateRequest,
)
from app.services.user_admin_service import UserAdminService

router = APIRouter(prefix="/users", tags=["users"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return _user_response(current_user)


@router.get("", response_model=UserListResponse)
def list_users(
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: UserAdminService = Depends(get_user_admin_service),
) -> UserListResponse:
    users = service.list_users()
    return UserListResponse(items=[_user_response(user) for user in users])


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    request: CreateUserRequest,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN)),
    service: UserAdminService = Depends(get_user_admin_service),
) -> UserResponse:
    try:
        role = UserRole(request.role)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role") from exc
    try:
        user = service.create_user(
            email=request.email,
            password=request.password,
            role=role,
            actor=current_user,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _user_response(user)


@router.patch("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    request: UpdateUserRequest,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN)),
    service: UserAdminService = Depends(get_user_admin_service),
) -> UserResponse:
    try:
        user = service.update_user(
            user_id=user_id,
            email=request.email,
            password=request.password,
            actor=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _user_response(user)


@router.patch("/{user_id}/status", response_model=UserResponse)
def update_user_status(
    user_id: str,
    request: UserStatusUpdateRequest,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN)),
    service: UserAdminService = Depends(get_user_admin_service),
) -> UserResponse:
    try:
        user = service.set_active(user_id=user_id, is_active=request.is_active, actor=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _user_response(user)


@router.patch("/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: str,
    request: UserRoleUpdateRequest,
    current_user: User = Depends(require_role(UserRole.SUPER_ADMIN)),
    service: UserAdminService = Depends(get_user_admin_service),
) -> UserResponse:
    try:
        role = UserRole(request.role)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role") from exc
    try:
        user = service.set_role(user_id=user_id, role=role, actor=current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _user_response(user)
