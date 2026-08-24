from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.domain.entities import (
    ChatConversation,
    ChatTurn,
    Lead,
    RecordStatus,
    SupportTicket,
    UserRole,
)
from app.main import app
from app.services.engagement_admin_service import LeadAdminService, SupportAdminService
from tests.unit.auth.helpers import as_user, make_user, user_repo


def _lead(**overrides) -> Lead:
    base = Lead(
        lead_id=str(uuid4()),
        name="Ada",
        phone="+919876543210",
        city="Noida",
        country="IN",
        product="TINY",
        conversation_id="conv-lead-1",
        status=RecordStatus.OPEN,
        created_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _ticket(**overrides) -> SupportTicket:
    base = SupportTicket(
        ticket_id=str(uuid4()),
        name="Ada",
        phone="+919876543210",
        product="TINY",
        issue="Not charging",
        conversation_id="conv-support-1",
        status=RecordStatus.OPEN,
        created_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _conversation(conversation_id: str) -> ChatConversation:
    return ChatConversation(
        conversation_id=conversation_id,
        turns=[
            ChatTurn(
                trace_id="trace-1",
                conversation_id=conversation_id,
                user_message="I want TINY",
                response="Happy to help with TINY.",
                created_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
            )
        ],
    )


class FakeLeadRepo:
    def __init__(self, lead: Lead) -> None:
        self.lead = lead

    def list_leads(self, *, limit: int, offset: int, status: RecordStatus | None = None):
        del limit, offset
        if status and self.lead.status != status:
            return []
        return [self.lead]

    def get_lead(self, lead_id: str):
        return self.lead if lead_id == self.lead.lead_id else None

    def update_lead_status(self, lead_id: str, status: RecordStatus):
        if lead_id != self.lead.lead_id:
            return None
        self.lead.status = status
        self.lead.updated_at = datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc)
        return self.lead


class FakeTicketRepo:
    def __init__(self, ticket: SupportTicket) -> None:
        self.ticket = ticket

    def list_tickets(self, *, limit: int, offset: int, status: RecordStatus | None = None):
        del limit, offset
        if status and self.ticket.status != status:
            return []
        return [self.ticket]

    def get_ticket(self, ticket_id: str):
        return self.ticket if ticket_id == self.ticket.ticket_id else None

    def update_ticket_status(self, ticket_id: str, status: RecordStatus):
        if ticket_id != self.ticket.ticket_id:
            return None
        self.ticket.status = status
        self.ticket.updated_at = datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc)
        return self.ticket


class FakeTraceRepo:
    def __init__(self, conversation: ChatConversation) -> None:
        self.conversation = conversation

    def get_conversation(self, conversation_id: str):
        return self.conversation if conversation_id == self.conversation.conversation_id else None


def test_leads_require_authentication() -> None:
    with user_repo([]):
        client = TestClient(app)
        assert client.get("/leads").status_code == 401


def test_user_cannot_list_leads() -> None:
    user, _ = make_user(role=UserRole.USER)
    with user_repo([user]), as_user(user):
        client = TestClient(app)
        assert client.get("/leads").status_code == 403


def test_admin_can_list_and_update_lead() -> None:
    admin, _ = make_user(role=UserRole.ADMIN)
    lead = _lead()
    conversation = _conversation(lead.conversation_id)
    from app.dependencies import get_lead_admin_service

    service = LeadAdminService(FakeLeadRepo(lead), FakeTraceRepo(conversation))
    app.dependency_overrides[get_lead_admin_service] = lambda: service
    try:
        with user_repo([admin]), as_user(admin):
            client = TestClient(app)
            listed = client.get("/leads")
            assert listed.status_code == 200
            item = listed.json()["items"][0]
            assert item["status"] == "open"
            assert item["id"] == lead.lead_id
            assert "conversation" not in item

            detail = client.get(f"/leads/{lead.lead_id}")
            assert detail.status_code == 200
            body = detail.json()
            assert body["country"] == "IN"
            assert body["conversation"][0]["role"] == "user"

            patched = client.patch(f"/leads/{lead.lead_id}", json={"status": "closed"})
            assert patched.status_code == 200
            assert patched.json()["status"] == "closed"
    finally:
        app.dependency_overrides.pop(get_lead_admin_service, None)


def test_super_admin_can_manage_leads() -> None:
    super_admin, _ = make_user(role=UserRole.SUPER_ADMIN)
    lead = _lead()
    from app.dependencies import get_lead_admin_service

    service = LeadAdminService(FakeLeadRepo(lead), FakeTraceRepo(_conversation(lead.conversation_id)))
    app.dependency_overrides[get_lead_admin_service] = lambda: service
    try:
        with user_repo([super_admin]), as_user(super_admin):
            client = TestClient(app)
            assert client.get("/leads").status_code == 200
            assert client.patch(f"/leads/{lead.lead_id}", json={"status": "closed"}).status_code == 200
    finally:
        app.dependency_overrides.pop(get_lead_admin_service, None)


def test_customer_service_admin_flow() -> None:
    admin, _ = make_user(role=UserRole.ADMIN)
    ticket = _ticket()
    conversation = _conversation(ticket.conversation_id)
    from app.dependencies import get_support_admin_service

    service = SupportAdminService(FakeTicketRepo(ticket), FakeTraceRepo(conversation))
    app.dependency_overrides[get_support_admin_service] = lambda: service
    try:
        with user_repo([admin]), as_user(admin):
            client = TestClient(app)
            listed = client.get("/customer-service")
            assert listed.status_code == 200
            item = listed.json()["items"][0]
            assert item["status"] == "open"
            assert item["issue"] == "Not charging"

            detail = client.get(f"/customer-service/{ticket.ticket_id}")
            assert detail.status_code == 200
            assert detail.json()["conversation"][1]["role"] == "assistant"

            patched = client.patch(
                f"/customer-service/{ticket.ticket_id}",
                json={"status": "closed"},
            )
            assert patched.status_code == 200
            assert patched.json()["status"] == "closed"
    finally:
        app.dependency_overrides.pop(get_support_admin_service, None)


def test_user_cannot_access_customer_service() -> None:
    user, _ = make_user(role=UserRole.USER)
    with user_repo([user]), as_user(user):
        client = TestClient(app)
        assert client.get("/customer-service").status_code == 403

def test_new_lead_defaults_to_open() -> None:
    lead = Lead(
        name="Ada",
        phone="+919876543210",
        city="Noida",
        country="IN",
        product="TINY",
        conversation_id="conv-1",
    )
    assert lead.status == RecordStatus.OPEN


def test_new_support_ticket_defaults_to_open() -> None:
    ticket = SupportTicket(
        name="Ada",
        phone="+919876543210",
        product="TINY",
        issue="Broken",
        conversation_id="conv-1",
    )
    assert ticket.status == RecordStatus.OPEN
