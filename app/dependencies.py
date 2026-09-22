import logging
from collections.abc import Callable
from functools import lru_cache
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.domain.entities import User, UserRole
from app.helpers.chat_auth import service_token_matches
from app.helpers.jwt_tokens import decode_access_token
from app.helpers.roles import role_at_least, role_in_allowed
from app.kb.ingestion.wiring import build_ingestion_service
from app.providers.llm.factory import get_llm_provider
from app.services.auth_service import AuthService, bootstrap_initial_super_admin
from app.services.chat_history import ChatHistoryService
from app.services.channels.intake import ChannelIntake
from app.services.conversation.orchestrator import ConversationOrchestrator
from app.services.generation.generation_service import GenerationService
from app.services.knowledge.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


@lru_cache
def get_ingestion_service() -> IngestionService:
    return build_ingestion_service()


@lru_cache
def get_chat_trace_repository():
    from app.db.engine import ensure_schema, get_session_factory
    from app.repositories.chat_trace import PostgresChatTraceRepository

    ensure_schema()
    return PostgresChatTraceRepository(get_session_factory())


@lru_cache
def get_background_chat_trace_repository():
    from app.repositories.chat_trace import BackgroundChatTraceRepository

    return BackgroundChatTraceRepository(get_chat_trace_repository())


@lru_cache
def get_chat_history_service() -> ChatHistoryService:
    return ChatHistoryService(get_chat_trace_repository())


@lru_cache
def get_orchestrator() -> ConversationOrchestrator:
    from app.db.engine import ensure_schema, get_session_factory
    from app.graph.factory import build_conversation_orchestrator
    from app.kb.retrieval.runtime_config import load_runtime_retrieval_config
    from app.kb.retrieval.service import RetrievalService
    from app.providers.knowledge.hybrid_knowledge_service import HybridKnowledgeService
    from app.providers.retrieval.factory import build_knowledge_hybrid_retriever
    from app.providers.crm import EarkartCrmClient
    from app.repositories.conversation import InMemoryConversationRepository
    from app.repositories.lead import PostgresLeadRepository
    from app.repositories.ticket import PostgresTicketRepository

    try:
        config = load_runtime_retrieval_config()
        vector_service = RetrievalService.from_config(config)
        hybrid = build_knowledge_hybrid_retriever(config, vector_service=vector_service)
        knowledge = HybridKnowledgeService(
            hybrid,
            retrieval_version=config.embedding_version,
        )
    except Exception:
        logger.exception("knowledge retriever initialization failed for production retrieval config")
        raise
    try:
        ensure_schema()
        sessions = get_session_factory()
        lead_tool = PostgresLeadRepository(sessions)
        ticket_tool = PostgresTicketRepository(sessions)
        traces = get_background_chat_trace_repository()
    except Exception:
        logger.exception("postgres business tables initialization failed")
        raise
    crm = None
    if settings.crm_lead_enabled:
        client = EarkartCrmClient()
        if client.is_configured:
            crm = client
        else:
            logger.warning("CRM lead sync enabled but CRM_LEAD_URL is not configured")
    return build_conversation_orchestrator(
        knowledge=knowledge,
        generation=GenerationService(
            get_llm_provider(settings),
            temperature=settings.generation_temperature,
            conversation_temperature=settings.generation_conversation_temperature,
            max_tokens=settings.generation_max_tokens,
        ),
        lead_tool=lead_tool,
        ticket_tool=ticket_tool,
        store=InMemoryConversationRepository(),
        model_used=settings.generation_model,
        traces=traces,
        crm=crm,
    )


@lru_cache
def get_user_repository():
    from app.db.engine import ensure_schema, get_session_factory
    from app.repositories.user import PostgresUserRepository

    ensure_schema()
    return PostgresUserRepository(get_session_factory())


def get_channel_intake(
    orchestrator: ConversationOrchestrator = Depends(get_orchestrator),
) -> ChannelIntake:
    return ChannelIntake(orchestrator)


@lru_cache
def get_auth_service() -> AuthService:
    users = get_user_repository()
    bootstrap_initial_super_admin(users)
    return AuthService(users)


@lru_cache
def get_user_admin_service():
    from app.services.user_admin_service import UserAdminService

    return UserAdminService(get_user_repository())


@lru_cache
def get_lead_admin_service():
    from app.db.engine import ensure_schema, get_session_factory
    from app.repositories.chat_trace import PostgresChatTraceRepository
    from app.repositories.lead import PostgresLeadRepository
    from app.services.engagement_admin_service import LeadAdminService

    ensure_schema()
    sessions = get_session_factory()
    return LeadAdminService(PostgresLeadRepository(sessions), PostgresChatTraceRepository(sessions))


@lru_cache
def get_support_admin_service():
    from app.db.engine import ensure_schema, get_session_factory
    from app.repositories.chat_trace import PostgresChatTraceRepository
    from app.repositories.ticket import PostgresTicketRepository
    from app.services.engagement_admin_service import SupportAdminService

    ensure_schema()
    sessions = get_session_factory()
    return SupportAdminService(PostgresTicketRepository(sessions), PostgresChatTraceRepository(sessions))


def _resolve_auth_service(request: Request) -> AuthService:
    override = request.app.dependency_overrides.get(get_auth_service)
    if override is not None:
        return override()
    return get_auth_service()


def require_chat_access(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    if service_token_matches(token, settings.chat_service_token):
        return
    try:
        decode_access_token(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    try:
        _resolve_auth_service(request).get_user_from_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    auth: AuthService = Depends(get_auth_service),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return auth.get_user_from_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_role(minimum_role: UserRole) -> Callable[..., User]:
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if not role_at_least(current_user.role, minimum_role):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return dependency


def require_roles(*allowed_roles: UserRole) -> Callable[..., User]:
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if not role_in_allowed(current_user.role, allowed_roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return dependency
