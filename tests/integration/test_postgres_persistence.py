from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from app.services.diagnostics.models import ChatTrace
from app.domain.entities import Lead, SupportTicket
from app.db.engine import ensure_schema, get_session_factory
from app.db.models import ChatTurnRow, LeadRow, RetrievalLayerHitRow, SupportTicketRow
from app.repositories.chat_trace import PostgresChatTraceRepository
from app.repositories.lead import PostgresLeadRepository
from app.repositories.ticket import PostgresTicketRepository


def _sessions():
    try:
        ensure_schema()
        return get_session_factory()
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")


@pytest.mark.integration
def test_postgres_lead_ticket_and_trace_persist() -> None:
    sessions = _sessions()
    conversation_id = f"test-{uuid.uuid4()}"
    leads = PostgresLeadRepository(sessions)
    tickets = PostgresTicketRepository(sessions)
    traces = PostgresChatTraceRepository(sessions)

    lead_id = leads.create_lead(
        Lead(
            name="Ada",
            phone="+919876543210",
            city="Noida",
            country="IN",
            product="TINY",
            conversation_id=conversation_id,
        )
    )
    ticket_id = tickets.create_support_ticket(
        SupportTicket(
            name="Ada",
            phone="+919876543210",
            product="TINY",
            issue="Not charging",
            conversation_id=conversation_id,
        )
    )

    trace = ChatTrace()
    trace.request = {
        "trace_id": f"tr-{uuid.uuid4()}",
        "conversation_id": conversation_id,
        "user_message": "What is TINY?",
    }
    trace.response = {"final_response": "TINY is compact."}
    trace.tool_execution = {
        "tool_name": "create_lead",
        "success": True,
        "lead_id": lead_id,
        "ticket_id": ticket_id,
    }
    trace.latency = {
        "retrieval_ms": 40.0,
        "vector_ms": 11.0,
        "keyword_ms": 9.0,
        "merge_ms": 1.0,
        "rerank_ms": 12.5,
        "threshold_ms": 0.4,
        "total_ms": 80.0,
    }
    trace.retrieval = {
        "backend": "llamaindex-hybrid",
        "vector": [
            {
                "rank": 1,
                "chunk_id": "chunk-tiny",
                "document_id": "doc-tiny",
                "vector_score": 0.9,
                "title": "TINY",
                "text_preview": "TINY is compact.",
            }
        ],
        "keyword": [],
        "merged": [],
    }
    traces.save(trace)

    listed = traces.list_conversations(limit=50, offset=0)
    brief = next(item for item in listed if item.conversation_id == conversation_id)
    assert brief.last_trace_id == trace.trace_id
    assert brief.turn_count == 1
    assert brief.last_total_ms == 80.0

    conversation = traces.get_conversation(conversation_id)
    assert conversation is not None
    assert conversation.turns[0].latency.vector_ms == 11.0
    assert conversation.turns[0].hits[0].chunk_id == "chunk-tiny"

    with sessions() as session:
        lead = session.execute(select(LeadRow).where(LeadRow.lead_id == lead_id)).scalar_one()
        ticket = session.execute(select(SupportTicketRow).where(SupportTicketRow.ticket_id == ticket_id)).scalar_one()
        turn = session.execute(select(ChatTurnRow).where(ChatTurnRow.trace_id == trace.trace_id)).scalar_one()
        hits = list(
            session.execute(
                select(RetrievalLayerHitRow).where(RetrievalLayerHitRow.trace_id == trace.trace_id)
            ).scalars()
        )
        assert lead.city == "Noida"
        assert ticket.issue == "Not charging"
        assert turn.vector_ms == 11.0
        assert turn.rerank_ms == 12.5
        assert turn.lead_id == lead_id
        assert turn.ticket_id == ticket_id
        assert hits[0].chunk_id == "chunk-tiny"
        assert hits[0].layer == "vector"
        session.delete(lead)
        session.delete(ticket)
        session.execute(delete(RetrievalLayerHitRow).where(RetrievalLayerHitRow.trace_id == trace.trace_id))
        session.delete(turn)
        session.commit()
