from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_lead_admin_service, require_role
from app.domain.entities import LeadRecordStatus, User, UserRole
from app.helpers.leads import lead_summary_payload
from app.schemas import LeadListResponse, LeadStatusUpdateRequest, LeadSummary
from app.services.lead_admin_service import LeadAdminService

router = APIRouter(prefix="/leads", tags=["leads"])


@router.get("", response_model=LeadListResponse)
def list_leads(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: LeadAdminService = Depends(get_lead_admin_service),
) -> LeadListResponse:
    items = service.list_leads(limit=limit, offset=offset)
    return LeadListResponse(items=[LeadSummary(**lead_summary_payload(item)) for item in items])


@router.get("/{lead_id}", response_model=LeadSummary)
def get_lead(
    lead_id: str,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: LeadAdminService = Depends(get_lead_admin_service),
) -> LeadSummary:
    lead = service.get_lead(lead_id)
    if lead is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    return LeadSummary(**lead_summary_payload(lead))


@router.patch("/{lead_id}/status", response_model=LeadSummary)
def update_lead_status(
    lead_id: str,
    request: LeadStatusUpdateRequest,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: LeadAdminService = Depends(get_lead_admin_service),
) -> LeadSummary:
    try:
        status_value = LeadRecordStatus(request.status)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status") from exc
    lead = service.update_status(lead_id, status_value)
    if lead is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    return LeadSummary(**lead_summary_payload(lead))
