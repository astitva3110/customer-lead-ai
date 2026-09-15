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


def _closed_by_label(user_id: str | None, user_labels: dict[str, str]) -> str | None:
    if not user_id:
        return None
    return user_labels.get(user_id)


def lead_list_payload(lead: Lead, *, user_labels: dict[str, str] | None = None) -> dict:
    labels = user_labels or {}
    return {
        "id": lead.lead_id,
        "name": lead.name,
        "phone": lead.phone,
        "city": lead.city,
        "product": lead.product,
        "status": lead.status.value,
        "closed_by": _closed_by_label(lead.closed_by, labels),
        "created_at": _iso(lead.created_at),
        "updated_at": _iso(lead.updated_at),
    }


def lead_detail_payload(
    lead: Lead,
    conversation: ChatConversation | None,
    *,
    user_labels: dict[str, str] | None = None,
) -> dict:
    return {
        **lead_list_payload(lead, user_labels=user_labels),
        "country": lead.country,
        "conversation_id": lead.conversation_id,
        "conversation": conversation_messages(conversation),
    }


def support_list_payload(ticket: SupportTicket, *, user_labels: dict[str, str] | None = None) -> dict:
    labels = user_labels or {}
    return {
        "id": ticket.ticket_id,
        "name": ticket.name,
        "phone": ticket.phone,
        "product": ticket.product,
        "issue": ticket.issue,
        "status": ticket.status.value,
        "closed_by": _closed_by_label(ticket.closed_by, labels),
        "created_at": _iso(ticket.created_at),
        "updated_at": _iso(ticket.updated_at),
    }


def support_detail_payload(
    ticket: SupportTicket,
    conversation: ChatConversation | None,
    *,
    user_labels: dict[str, str] | None = None,
) -> dict:
    return {
        **support_list_payload(ticket, user_labels=user_labels),
        "conversation_id": ticket.conversation_id,
        "conversation": conversation_messages(conversation),
    }


def closed_by_user_labels(
    records: list[Lead] | list[SupportTicket],
    users: object,
) -> dict[str, str]:
    user_ids = {record.closed_by for record in records if record.closed_by}
    if not user_ids:
        return {}
    return users.get_emails_by_ids(user_ids)
