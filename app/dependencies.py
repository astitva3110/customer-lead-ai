import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.domain.entities import User, UserRole
from app.helpers.roles import role_at_least, role_in_allowed
from app.kb.ingestion.wiring import build_ingestion_service
from app.providers.llm.factory import get_llm_provider
from app.services.auth_service import AuthService, bootstrap_initial_super_admin
from app.services.chat_history import ChatHistoryService
from app.services.conversation.orchestrator import ConversationOrchestrator
from app.services.generation.generation_service import GenerationService
from app.services.knowledge.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

_RETRIEVAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "retrieval" / "v2.yaml"
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
    from app.kb.evaluation.retrieval_config import RetrievalConfig
    from app.kb.retrieval.service import RetrievalService
    from app.providers.knowledge.knowledge_service import LlamaIndexKnowledgeService
    from app.providers.retrieval.factory import build_knowledge_hybrid_retriever
    from app.repositories.conversation import InMemoryConversationRepository
    from app.repositories.lead import PostgresLeadRepository
    from app.repositories.ticket import PostgresTicketRepository

    try:
        config = RetrievalConfig.from_yaml(_RETRIEVAL_CONFIG)
        vector_service = RetrievalService.from_config(config)
        hybrid = build_knowledge_hybrid_retriever(config, vector_service=vector_service)
        knowledge = LlamaIndexKnowledgeService(
            hybrid,
            retrieval_version=config.embedding_version,
        )
    except Exception:
        logger.exception("knowledge retriever initialization failed from %s", _RETRIEVAL_CONFIG)
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
    )


@lru_cache
def get_user_repository():
    from app.db.engine import ensure_schema, get_session_factory
    from app.repositories.user import PostgresUserRepository

    ensure_schema()
    return PostgresUserRepository(get_session_factory())


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
    from app.repositories.lead import PostgresLeadRepository
    from app.services.lead_admin_service import LeadAdminService

    ensure_schema()
    return LeadAdminService(PostgresLeadRepository(get_session_factory()))


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
