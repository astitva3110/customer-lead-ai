from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.services.conversation.orchestrator import ConversationOrchestrator
from app.services.diagnostics.recorder import tracing_enabled
from app.services.chat_history import ChatHistoryService
from app.helpers.chat_history import conversation_brief_payload, conversation_detail_payload
from app.dependencies import get_chat_history_service, get_orchestrator, require_role
from app.domain.entities import User, UserRole
from app.schemas import (
    ChatConversationSummary,
    ChatDetailResponse,
    ChatListResponse,
    ChatRequest,
    ChatResponse,
    SourceChunk,
)

router = APIRouter(tags=["chat"])
_CHAT_HTML = Path(__file__).resolve().parents[1] / "static" / "chat.html"


@router.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(_CHAT_HTML, media_type="text/html")


@router.get("/", include_in_schema=False)
def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat")


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(
    request: ChatRequest,
    orchestrator: ConversationOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    result = orchestrator.handle(request.conversation_id, request.message, country=request.country)
    payload: dict = {
        "conversation_id": result.conversation_id,
        "mode": result.mode or "KNOWLEDGE",
        "response": result.response,
        "answer": result.response,
        "sources": [
            SourceChunk(
                url=str(source.get("url") or ""),
                title=str(source.get("title") or ""),
                score=float(source.get("score") or 0.0),
            )
            for source in result.sources
        ],
    }
    trace_id = (result.trace or {}).get("trace_id")
    if trace_id and tracing_enabled():
        payload["debug_trace_id"] = trace_id
    return ChatResponse(**payload)


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
