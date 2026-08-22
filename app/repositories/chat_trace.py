from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import aliased, sessionmaker

from app.services.diagnostics.models import ChatTrace
from app.domain.entities import ChatConversation, ChatConversationBrief
from app.helpers.chat_trace_persist import chat_turn_fields, retrieval_layer_hit_fields
from app.db.engine import get_session_factory
from app.db.models import ChatTurnRow, RetrievalLayerHitRow
from app.db.trace_queue import submit_trace_write


class PostgresChatTraceRepository:
    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def save(self, trace: ChatTrace) -> None:
        fields = chat_turn_fields(trace)
        if not fields["trace_id"]:
            return
        hits = retrieval_layer_hit_fields(trace)
        with self._session_factory() as session:
            session.merge(ChatTurnRow(**fields))
            session.execute(delete(RetrievalLayerHitRow).where(RetrievalLayerHitRow.trace_id == fields["trace_id"]))
            for item in hits:
                session.add(RetrievalLayerHitRow(**item))
            session.commit()

    def list_conversations(self, *, limit: int, offset: int) -> list[ChatConversationBrief]:
        ranked = (
            select(
                ChatTurnRow,
                func.row_number()
                .over(
                    partition_by=ChatTurnRow.conversation_id,
                    order_by=ChatTurnRow.created_at.desc(),
                )
                .label("rn"),
                func.count().over(partition_by=ChatTurnRow.conversation_id).label("turn_count"),
            )
        ).subquery()
        latest = aliased(ChatTurnRow, ranked)
        with self._session_factory() as session:
            rows = session.execute(
                select(latest, ranked.c.turn_count)
                .where(ranked.c.rn == 1)
                .order_by(latest.created_at.desc())
                .limit(limit)
                .offset(offset)
            ).all()
        return [turn.to_brief(int(count)) for turn, count in rows]

    def get_conversation(self, conversation_id: str) -> ChatConversation | None:
        with self._session_factory() as session:
            turns = list(
                session.execute(
                    select(ChatTurnRow)
                    .where(ChatTurnRow.conversation_id == conversation_id)
                    .order_by(ChatTurnRow.created_at.asc())
                ).scalars()
            )
            if not turns:
                return None
            hits = list(
                session.execute(
                    select(RetrievalLayerHitRow)
                    .where(RetrievalLayerHitRow.trace_id.in_([row.trace_id for row in turns]))
                    .order_by(RetrievalLayerHitRow.layer, RetrievalLayerHitRow.rank)
                ).scalars()
            )
        by_trace: dict[str, list] = {}
        for hit in hits:
            by_trace.setdefault(hit.trace_id, []).append(hit.to_entity())
        return ChatConversation(
            conversation_id=conversation_id,
            turns=[row.to_entity(by_trace.get(row.trace_id, [])) for row in turns],
        )


class BackgroundChatTraceRepository:
    """Same read/write API; save() never waits on Postgres."""

    def __init__(self, inner: PostgresChatTraceRepository) -> None:
        self._inner = inner

    def save(self, trace: ChatTrace) -> None:
        submit_trace_write(self._inner.save, trace)

    def list_conversations(self, *, limit: int, offset: int) -> list[ChatConversationBrief]:
        return self._inner.list_conversations(limit=limit, offset=offset)

    def get_conversation(self, conversation_id: str) -> ChatConversation | None:
        return self._inner.get_conversation(conversation_id)
