from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.config import settings
from app.helpers.rate_limit import limiter
from app.services.conversation.orchestrator import ConversationOrchestrator
from app.services.chat_history import ChatHistoryService
from app.helpers.chat_api import chat_result_payload
from app.helpers.chat_history import conversation_brief_payload, conversation_detail_payload
from app.dependencies import get_chat_history_service, get_orchestrator, require_chat_access, require_role
from app.domain.entities import User, UserRole
from app.schemas import (
    ChatConversationSummary,
    ChatDetailResponse,
    ChatListResponse,
    ChatRequest,
    ChatResponse,
)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
@limiter.limit(settings.rate_limit_chat)
def chat(
    request: Request,
    chat_request: ChatRequest,
    _auth: None = Depends(require_chat_access),
    orchestrator: ConversationOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    result = orchestrator.handle(
        chat_request.conversation_id,
        chat_request.message,
        country=chat_request.country,
        channel=chat_request.channel,
        origin=chat_request.origin,
        source=chat_request.source,
        phone=chat_request.phone,
        user_name=chat_request.user_name,
    )
    return ChatResponse(**chat_result_payload(result))


@router.get("/chats", response_model=ChatListResponse)
def list_chats(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatListResponse:
    items = service.list_conversations(limit=limit, offset=offset)
    return ChatListResponse(items=[ChatConversationSummary(**conversation_brief_payload(item)) for item in items])


@router.get("/chats/{conversation_id}", response_model=ChatDetailResponse)
def get_chat(
    conversation_id: str,
    _current_user: User = Depends(require_role(UserRole.ADMIN)),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatDetailResponse:
    conversation = service.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatDetailResponse(**conversation_detail_payload(conversation))
