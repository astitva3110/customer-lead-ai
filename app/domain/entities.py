"""Domain entities. No SQLAlchemy, no FastAPI, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class RecordStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


# Backward-compatible alias for existing imports during migration.
LeadRecordStatus = RecordStatus


@dataclass
class User:
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    user_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Lead:
    name: str
    phone: str
    city: str
    country: str
    product: str
    conversation_id: str
    lead_id: str = field(default_factory=lambda: str(uuid4()))
    status: RecordStatus = RecordStatus.OPEN
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class SupportTicket:
    name: str
    phone: str
    product: str
    issue: str
    conversation_id: str
    ticket_id: str = field(default_factory=lambda: str(uuid4()))
    status: RecordStatus = RecordStatus.OPEN
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class ChatTurnLatency:
    guardrail_ms: float | None = None
    routing_ms: float | None = None
    rewrite_ms: float | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    tool_ms: float | None = None
    total_ms: float | None = None
    vector_ms: float | None = None
    keyword_ms: float | None = None
    merge_ms: float | None = None
    rerank_ms: float | None = None
    threshold_ms: float | None = None


@dataclass
class RetrievalLayerHit:
    layer: str
    rank: int
    chunk_id: str
    document_id: str = ""
    title: str = ""
    section_path: str = ""
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    combined_score: float | None = None
    original_retrieval_rank: int | None = None
    text_preview: str = ""


@dataclass
class ChatTurn:
    trace_id: str
    conversation_id: str
    user_message: str = ""
    rewritten_query: str | None = None
    intent: str = ""
    response: str = ""
    tool_name: str | None = None
    tool_success: bool | None = None
    lead_id: str | None = None
    ticket_id: str | None = None
    backend: str | None = None
    reranker_name: str | None = None
    latency: ChatTurnLatency = field(default_factory=ChatTurnLatency)
    created_at: datetime | None = None
    hits: list[RetrievalLayerHit] = field(default_factory=list)


@dataclass
class ChatConversationBrief:
    conversation_id: str
    last_trace_id: str
    last_user_message: str
    last_response: str
    last_intent: str
    last_total_ms: float | None
    turn_count: int
    updated_at: datetime | None = None


@dataclass
class ChatConversation:
    conversation_id: str
    turns: list[ChatTurn]
