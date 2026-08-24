from __future__ import annotations

from datetime import datetime

from app.domain.entities import ChatConversation, Lead, SupportTicket


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def conversation_messages(conversation: ChatConversation | None) -> list[dict]:
    if conversation is None:
        return []
    messages: list[dict] = []
    for turn in conversation.turns:
        if turn.user_message:
            messages.append(
                {
                    "role": "user",
                    "content": turn.user_message,
                    "created_at": _iso(turn.created_at),
                }
            )
        if turn.response:
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.response,
                    "created_at": _iso(turn.created_at),
                }
            )
    return messages


def lead_list_payload(lead: Lead) -> dict:
    return {
        "id": lead.lead_id,
        "name": lead.name,
        "phone": lead.phone,
        "city": lead.city,
        "product": lead.product,
        "status": lead.status.value,
        "created_at": _iso(lead.created_at),
        "updated_at": _iso(lead.updated_at),
    }


def lead_detail_payload(lead: Lead, conversation: ChatConversation | None) -> dict:
    return {
        **lead_list_payload(lead),
        "country": lead.country,
        "conversation_id": lead.conversation_id,
        "conversation": conversation_messages(conversation),
    }


def support_list_payload(ticket: SupportTicket) -> dict:
    return {
        "id": ticket.ticket_id,
        "name": ticket.name,
        "phone": ticket.phone,
        "product": ticket.product,
        "issue": ticket.issue,
        "status": ticket.status.value,
        "created_at": _iso(ticket.created_at),
        "updated_at": _iso(ticket.updated_at),
    }


def support_detail_payload(ticket: SupportTicket, conversation: ChatConversation | None) -> dict:
    return {
        **support_list_payload(ticket),
        "conversation_id": ticket.conversation_id,
        "conversation": conversation_messages(conversation),
    }
