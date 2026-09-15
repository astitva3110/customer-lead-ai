from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.services.conversation.models import (
    ConversationGoal,
    ConversationState,
    LeadStage,
    LeadStatus,
    LeadWorkflow,
    SupportWorkflow,
    TicketStatus,
    TurnIntent,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_activity_at(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def session_idle_seconds(state: ConversationState, *, now: datetime | None = None) -> float | None:
    last_active = _parse_activity_at(state.last_activity_at)
    if last_active is None:
        return None
    current = now or _utc_now()
    return max(0.0, (current - last_active).total_seconds())


def reset_conversation_session(state: ConversationState) -> None:
    """Drop stale turn/workflow context while keeping channel identity."""
    state.conversation_history = []
    state.intent = ""
    state.mode = ""
    state.return_mode = ""
    state.awaiting_field = ""
    state.conversation_goal = ConversationGoal.NONE
    state.current_turn_intent = TurnIntent.UNKNOWN
    state.lead_workflow = LeadWorkflow.NONE
    state.support_workflow = SupportWorkflow.NONE
    state.lead_intent = False
    state.support_intent = False
    state.sales_interest = False
    state.lead_stage = LeadStage.NOT_STARTED
    state.lead_collection_active = False
    state.support_collection_active = False
    state.explicit_action = ""
    state.user_context = {}
    state.product = ""
    state.product_id = ""
    state.city = ""
    state.pending_phone = ""
    state.support_issue = ""
    state.retrieved_context = []
    state.retrieved_chunk_ids = []
    state.retrieval_metadata = {}
    state.lead_status = LeadStatus.IDLE
    state.ticket_status = TicketStatus.IDLE
    state.response = ""
    state.sources = []
    state.query_rewritten = ""
    state.guardrail_rejected = False
    state.error = ""
    state.trace = {}


def apply_idle_session_timeout(
    state: ConversationState,
    timeout_minutes: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Reset stale session state after idle timeout. Returns True when reset."""
    timeout = int(timeout_minutes or 0)
    if timeout <= 0:
        return False
    idle_seconds = session_idle_seconds(state, now=now)
    if idle_seconds is None:
        return False
    if idle_seconds < timeout * 60:
        return False
    logger.info(
        "conversation session expired conversation_id=%s idle_seconds=%.0f timeout_minutes=%s",
        state.conversation_id,
        idle_seconds,
        timeout,
    )
    reset_conversation_session(state)
    return True


def touch_session_activity(state: ConversationState, *, now: datetime | None = None) -> None:
    current = now or _utc_now()
    state.last_activity_at = current.isoformat()


def activity_payload(state: ConversationState) -> dict[str, Any]:
    idle_seconds = session_idle_seconds(state)
    return {
        "last_activity_at": state.last_activity_at or "",
        "idle_seconds": round(idle_seconds, 3) if idle_seconds is not None else None,
    }
