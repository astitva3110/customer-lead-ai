from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_lead_admin_service, require_role
from app.domain.entities import RecordStatus, User, UserRole
from app.helpers.engagement import lead_detail_payload, lead_list_payload
from app.schemas import LeadDetail, LeadListResponse, LeadSummary, LeadUpdateRequest
from app.services.engagement_admin_service import LeadAdminService

router = APIRouter(prefix="/leads", tags=["leads"])


def _parse_status(value: str) -> RecordStatus:
    try:
        return RecordStatus(value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status") from exc


@router.get("", response_model=LeadListResponse)
def list_leads(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: LeadAdminService = Depends(get_lead_admin_service),
) -> LeadListResponse:
    status_value = _parse_status(status_filter) if status_filter else None
    items = service.list_leads(limit=limit, offset=offset, status=status_value)
    return LeadListResponse(items=[LeadSummary(**lead_list_payload(item)) for item in items])


@router.get("/{lead_id}", response_model=LeadDetail)
def get_lead(
    lead_id: str,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: LeadAdminService = Depends(get_lead_admin_service),
) -> LeadDetail:
    detail = service.get_lead_detail(lead_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    lead, conversation = detail
    return LeadDetail(**lead_detail_payload(lead, conversation))


@router.patch("/{lead_id}", response_model=LeadDetail)
def update_lead(
    lead_id: str,
    request: LeadUpdateRequest,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: LeadAdminService = Depends(get_lead_admin_service),
) -> LeadDetail:
    status_value = _parse_status(request.status)
    lead = service.update_status(lead_id, status_value)
    if lead is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    detail = service.get_lead_detail(lead_id)
    conversation = detail[1] if detail else None
    return LeadDetail(**lead_detail_payload(lead, conversation))
