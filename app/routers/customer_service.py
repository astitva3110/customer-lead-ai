from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_support_admin_service, require_role
from app.domain.entities import RecordStatus, User, UserRole
from app.helpers.engagement import support_detail_payload, support_list_payload
from app.schemas import SupportDetail, SupportListResponse, SupportSummary, SupportUpdateRequest
from app.services.engagement_admin_service import SupportAdminService

router = APIRouter(prefix="/customer-service", tags=["customer-service"])


def _parse_status(value: str) -> RecordStatus:
    try:
        return RecordStatus(value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status") from exc


@router.get("", response_model=SupportListResponse)
def list_customer_service(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: SupportAdminService = Depends(get_support_admin_service),
) -> SupportListResponse:
    status_value = _parse_status(status_filter) if status_filter else None
    items = service.list_tickets(limit=limit, offset=offset, status=status_value)
    return SupportListResponse(items=[SupportSummary(**support_list_payload(item)) for item in items])


@router.get("/{ticket_id}", response_model=SupportDetail)
def get_customer_service(
    ticket_id: str,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: SupportAdminService = Depends(get_support_admin_service),
) -> SupportDetail:
    detail = service.get_ticket_detail(ticket_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Support request not found")
    ticket, conversation = detail
    return SupportDetail(**support_detail_payload(ticket, conversation))


@router.patch("/{ticket_id}", response_model=SupportDetail)
def update_customer_service(
    ticket_id: str,
    request: SupportUpdateRequest,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: SupportAdminService = Depends(get_support_admin_service),
) -> SupportDetail:
    status_value = _parse_status(request.status)
    ticket = service.update_status(ticket_id, status_value)
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Support request not found")
    detail = service.get_ticket_detail(ticket_id)
    conversation = detail[1] if detail else None
    return SupportDetail(**support_detail_payload(ticket, conversation))
