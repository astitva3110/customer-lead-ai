from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi.testclient import TestClient

from app.config import settings
from app.domain.entities import Lead, LeadRecordStatus, UserRole
from app.helpers.jwt_tokens import create_access_token
from app.main import app
from app.repositories.memory_user import InMemoryUserRepository
from app.services.auth_service import AuthService
from tests.unit.auth.helpers import as_user, auth_header, make_user, user_repo


def test_login_valid_credentials() -> None:
    user, password = make_user(role=UserRole.USER)
    with user_repo([user]):
        client = TestClient(app)
        response = client.post("/auth/login", json={"email": user.email, "password": password})
        assert response.status_code == 200
        assert response.json()["access_token"]
        assert response.json()["token_type"] == "bearer"


def test_login_invalid_password() -> None:
    user, _password = make_user(role=UserRole.USER)
    with user_repo([user]):
        client = TestClient(app)
        response = client.post("/auth/login", json={"email": user.email, "password": "wrong-pass"})
        assert response.status_code == 401


def test_login_unknown_user() -> None:
    with user_repo([]):
        client = TestClient(app)
        response = client.post("/auth/login", json={"email": "missing@example.com", "password": "password123"})
        assert response.status_code == 401


def test_login_disabled_user() -> None:
    user, password = make_user(role=UserRole.USER, is_active=False)
    with user_repo([user]):
        client = TestClient(app)
        response = client.post("/auth/login", json={"email": user.email, "password": password})
        assert response.status_code == 401


def test_invalid_token_rejected() -> None:
    with user_repo([]):
        client = TestClient(app)
        response = client.get("/users/me", headers={"Authorization": "Bearer invalid-token"})
        assert response.status_code == 401


def test_expired_token_rejected() -> None:
    user, _password = make_user(role=UserRole.USER)
    expired = jwt.encode(
        {
            "sub": user.user_id,
            "role": user.role.value,
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    repo = InMemoryUserRepository()
    repo.create_user(user)
    with user_repo([user]):
        client = TestClient(app)
        response = client.get("/users/me", headers={"Authorization": f"Bearer {expired}"})
        assert response.status_code == 401


def test_disabled_user_token_rejected() -> None:
    user, _password = make_user(role=UserRole.USER, is_active=False)
    with user_repo([user]):
        client = TestClient(app)
        response = client.get("/users/me", headers=auth_header(user))
        assert response.status_code == 401


def test_user_can_access_me_and_chat() -> None:
    user, _password = make_user(role=UserRole.USER)
    with user_repo([user]), as_user(user):
        client = TestClient(app)
        assert client.get("/users/me", headers=auth_header(user)).status_code == 200


def test_user_cannot_access_admin_chat_history() -> None:
    user, _password = make_user(role=UserRole.USER)
    with user_repo([user]), as_user(user):
        client = TestClient(app)
        assert client.get("/chats").status_code == 403


def test_user_cannot_access_super_admin_user_creation() -> None:
    user, _password = make_user(role=UserRole.USER)
    with user_repo([user]), as_user(user):
        client = TestClient(app)
        response = client.post(
            "/users",
            json={"email": "new@example.com", "password": "password123", "role": "user"},
        )
        assert response.status_code == 403


def test_admin_can_access_chat_history() -> None:
    admin, _password = make_user(role=UserRole.ADMIN)

    class FakeTraceRepo:
        def list_conversations(self, *, limit: int, offset: int):
            return []

    from app.dependencies import get_chat_history_service
    from app.services.chat_history import ChatHistoryService

    app.dependency_overrides[get_chat_history_service] = lambda: ChatHistoryService(FakeTraceRepo())
    try:
        with user_repo([admin]), as_user(admin):
            client = TestClient(app)
            assert client.get("/chats").status_code == 200
    finally:
        app.dependency_overrides.pop(get_chat_history_service, None)


def test_admin_can_view_leads(monkeypatch) -> None:
    admin, _password = make_user(role=UserRole.ADMIN)
    lead = Lead(
        lead_id=str(uuid4()),
        name="Ada",
        phone="+919876543210",
        city="Noida",
        country="IN",
        product="TINY",
        conversation_id="conv-1",
        status=LeadRecordStatus.NEW,
    )

    class FakeLeadRepo:
        def list_leads(self, *, limit: int, offset: int) -> list[Lead]:
            return [lead]

        def get_lead(self, lead_id: str) -> Lead | None:
            return lead if lead_id == lead.lead_id else None

        def update_lead_status(self, lead_id: str, status: LeadRecordStatus) -> Lead | None:
            if lead_id != lead.lead_id:
                return None
            lead.status = status
            return lead

    from app.dependencies import get_lead_admin_service
    from app.services.lead_admin_service import LeadAdminService

    app.dependency_overrides[get_lead_admin_service] = lambda: LeadAdminService(FakeLeadRepo())  # type: ignore[arg-type]
    try:
        with user_repo([admin]), as_user(admin):
            client = TestClient(app)
            assert client.get("/leads").status_code == 200
            assert client.get(f"/leads/{lead.lead_id}").status_code == 200
            patch = client.patch(f"/leads/{lead.lead_id}/status", json={"status": "contacted"})
            assert patch.status_code == 200
            assert patch.json()["status"] == "contacted"
    finally:
        app.dependency_overrides.pop(get_lead_admin_service, None)


def test_admin_cannot_manage_super_admin_role() -> None:
    admin, _password = make_user(role=UserRole.ADMIN)
    target, _ = make_user(role=UserRole.USER)
    with user_repo([admin, target]), as_user(admin):
        client = TestClient(app)
        response = client.patch(f"/users/{target.user_id}/role", json={"role": "super_admin"})
        assert response.status_code == 403


def test_super_admin_can_create_users() -> None:
    super_admin, _password = make_user(role=UserRole.SUPER_ADMIN)
    with user_repo([super_admin]), as_user(super_admin):
        client = TestClient(app)
        response = client.post(
            "/users",
            json={"email": "created@example.com", "password": "password123", "role": "user"},
        )
        assert response.status_code == 201
        assert response.json()["email"] == "created@example.com"


def test_super_admin_can_create_admin() -> None:
    super_admin, _password = make_user(role=UserRole.SUPER_ADMIN)
    with user_repo([super_admin]), as_user(super_admin):
        client = TestClient(app)
        response = client.post(
            "/users",
            json={"email": "admin@example.com", "password": "password123", "role": "admin"},
        )
        assert response.status_code == 201
        assert response.json()["role"] == "admin"


def test_super_admin_can_change_roles_and_deactivate() -> None:
    super_admin, _password = make_user(role=UserRole.SUPER_ADMIN)
    target, _ = make_user(role=UserRole.USER)
    with user_repo([super_admin, target]), as_user(super_admin):
        client = TestClient(app)
        role_response = client.patch(f"/users/{target.user_id}/role", json={"role": "admin"})
        assert role_response.status_code == 200
        assert role_response.json()["role"] == "admin"
        status_response = client.patch(f"/users/{target.user_id}/status", json={"is_active": False})
        assert status_response.status_code == 200
        assert status_response.json()["is_active"] is False


def test_super_admin_inherits_admin_endpoints() -> None:
    super_admin, _password = make_user(role=UserRole.SUPER_ADMIN)

    class FakeTraceRepo:
        def list_conversations(self, *, limit: int, offset: int):
            return []

    from app.dependencies import get_chat_history_service
    from app.services.chat_history import ChatHistoryService

    app.dependency_overrides[get_chat_history_service] = lambda: ChatHistoryService(FakeTraceRepo())
    try:
        with user_repo([super_admin]), as_user(super_admin):
            client = TestClient(app)
            assert client.get("/chats").status_code == 200
    finally:
        app.dependency_overrides.pop(get_chat_history_service, None)


def test_admin_cannot_access_super_admin_only_endpoints() -> None:
    admin, _password = make_user(role=UserRole.ADMIN)
    with user_repo([admin]), as_user(admin):
        client = TestClient(app)
        response = client.post(
            "/users",
            json={"email": "blocked@example.com", "password": "password123", "role": "user"},
        )
        assert response.status_code == 403


def test_auth_service_get_user_from_token() -> None:
    user, _password = make_user(role=UserRole.USER)
    repo = InMemoryUserRepository()
    repo.create_user(user)
    auth = AuthService(repo)
    token = create_access_token(user_id=user.user_id, role=user.role)
    resolved = auth.get_user_from_token(token)
    assert resolved.user_id == user.user_id
