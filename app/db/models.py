"""SQLAlchemy tables and mapping to domain entities. Persistence only."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.domain.entities import (
    ChatConversationBrief,
    ChatTurn,
    ChatTurnLatency,
    Lead,
    RecordStatus,
    RetrievalLayerHit,
    SupportTicket,
    User,
    UserRole,
)


class AppBase(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LeadRow(AppBase):
    __tablename__ = "leads"

    lead_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    phone: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str] = mapped_column(Text, nullable=False)
    product: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=RecordStatus.OPEN.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    def to_entity(self) -> Lead:
        return Lead(
            lead_id=self.lead_id,
            name=self.name,
            phone=self.phone,
            city=self.city,
            country=self.country,
            product=self.product,
            conversation_id=self.conversation_id,
            status=RecordStatus(self.status),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_entity(cls, lead: Lead) -> LeadRow:
        return cls(
            lead_id=lead.lead_id,
            name=lead.name,
            phone=lead.phone,
            city=lead.city,
            country=lead.country,
            product=lead.product,
            conversation_id=lead.conversation_id,
            status=lead.status.value,
        )


class UserRow(AppBase):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default=UserRole.USER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    def to_entity(self) -> User:
        return User(
            user_id=self.user_id,
            email=self.email,
            password_hash=self.password_hash,
            role=UserRole(self.role),
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_entity(cls, user: User) -> UserRow:
        return cls(
            user_id=user.user_id,
            email=user.email.lower(),
            password_hash=user.password_hash,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at or _utcnow(),
            updated_at=user.updated_at or _utcnow(),
        )


class SupportTicketRow(AppBase):
    __tablename__ = "support_tickets"

    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    phone: Mapped[str] = mapped_column(Text, nullable=False)
    product: Mapped[str] = mapped_column(Text, nullable=False)
    issue: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=RecordStatus.OPEN.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    def to_entity(self) -> SupportTicket:
        return SupportTicket(
            ticket_id=self.ticket_id,
            name=self.name,
            phone=self.phone,
            product=self.product,
            issue=self.issue,
            conversation_id=self.conversation_id,
            status=RecordStatus(self.status),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_entity(cls, ticket: SupportTicket) -> SupportTicketRow:
        return cls(
            ticket_id=ticket.ticket_id,
            name=ticket.name,
            phone=ticket.phone,
            product=ticket.product,
            issue=ticket.issue,
            conversation_id=ticket.conversation_id,
            status=ticket.status.value,
        )


class ChatTurnRow(AppBase):
    __tablename__ = "chat_turns"

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    response: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    lead_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ticket_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reranker_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    guardrail_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    routing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rewrite_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tool_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    vector_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    merge_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    def to_entity(self, hits: list[RetrievalLayerHit] | None = None) -> ChatTurn:
        return ChatTurn(
            trace_id=self.trace_id,
            conversation_id=self.conversation_id,
            user_message=self.user_message,
            rewritten_query=self.rewritten_query,
            intent=self.intent,
            response=self.response,
            tool_name=self.tool_name,
            tool_success=self.tool_success,
            lead_id=self.lead_id,
            ticket_id=self.ticket_id,
            backend=self.backend,
            reranker_name=self.reranker_name,
            latency=ChatTurnLatency(
                guardrail_ms=self.guardrail_ms,
                routing_ms=self.routing_ms,
                rewrite_ms=self.rewrite_ms,
                retrieval_ms=self.retrieval_ms,
                generation_ms=self.generation_ms,
                tool_ms=self.tool_ms,
                total_ms=self.total_ms,
                vector_ms=self.vector_ms,
                keyword_ms=self.keyword_ms,
                merge_ms=self.merge_ms,
                rerank_ms=self.rerank_ms,
                threshold_ms=self.threshold_ms,
            ),
            created_at=self.created_at,
            hits=list(hits or []),
        )

    def to_brief(self, turn_count: int) -> ChatConversationBrief:
        return ChatConversationBrief(
            conversation_id=self.conversation_id,
            last_trace_id=self.trace_id,
            last_user_message=self.user_message,
            last_response=self.response,
            last_intent=self.intent,
            last_total_ms=self.total_ms,
            turn_count=turn_count,
            updated_at=self.created_at,
        )


class RetrievalLayerHitRow(AppBase):
    __tablename__ = "retrieval_layer_hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    layer: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    section_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    vector_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    combined_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    original_retrieval_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    def to_entity(self) -> RetrievalLayerHit:
        return RetrievalLayerHit(
            layer=self.layer,
            rank=self.rank,
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            title=self.title,
            section_path=self.section_path,
            vector_score=self.vector_score,
            keyword_score=self.keyword_score,
            rerank_score=self.rerank_score,
            combined_score=self.combined_score,
            original_retrieval_rank=self.original_retrieval_rank,
            text_preview=self.text_preview,
        )
