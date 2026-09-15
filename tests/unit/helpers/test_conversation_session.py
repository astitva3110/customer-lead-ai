from datetime import datetime, timedelta, timezone

from app.helpers.conversation_session import (
    apply_idle_session_timeout,
    reset_conversation_session,
    session_idle_seconds,
    touch_session_activity,
)
from app.services.conversation.models import ConversationGoal, ConversationState, LeadWorkflow


def test_reset_conversation_session_clears_workflow_but_keeps_identity() -> None:
    state = ConversationState(
        conversation_id="whatsapp:+919876543210",
        channel="whatsapp",
        origin="whatsapp",
        phone="+919876543210",
        user_name="Ada",
        conversation_history=[{"role": "user", "content": "Hi"}],
        product="TINY",
        conversation_goal=ConversationGoal.LEAD,
        lead_workflow=LeadWorkflow.DISCUSSING_PRODUCT,
    )
    reset_conversation_session(state)
    assert state.conversation_history == []
    assert state.product == ""
    assert state.conversation_goal == ConversationGoal.NONE
    assert state.lead_workflow == LeadWorkflow.NONE
    assert state.phone == "+919876543210"
    assert state.user_name == "Ada"


def test_apply_idle_session_timeout_resets_after_limit() -> None:
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    state = ConversationState(
        conversation_id="whatsapp:+919876543210",
        channel="whatsapp",
        conversation_history=[{"role": "user", "content": "old"}],
        product="TINY",
        last_activity_at=(now - timedelta(minutes=16)).isoformat(),
    )
    expired = apply_idle_session_timeout(state, 15, now=now + timedelta(minutes=1))
    assert expired is True
    assert state.conversation_history == []
    assert state.product == ""


def test_apply_idle_session_timeout_keeps_active_session() -> None:
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    state = ConversationState(
        conversation_id="whatsapp:+919876543210",
        conversation_history=[{"role": "user", "content": "recent"}],
        product="TINY",
        last_activity_at=(now - timedelta(minutes=5)).isoformat(),
    )
    expired = apply_idle_session_timeout(state, 15, now=now + timedelta(minutes=5))
    assert expired is False
    assert state.product == "TINY"
    assert state.conversation_history


def test_apply_idle_session_timeout_disabled_when_zero() -> None:
    state = ConversationState(
        conversation_id="web-1",
        last_activity_at="2020-01-01T00:00:00+00:00",
        product="TINY",
    )
    assert apply_idle_session_timeout(state, 0) is False
    assert state.product == "TINY"


def test_touch_session_activity_sets_timestamp() -> None:
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    state = ConversationState(conversation_id="web-1")
    touch_session_activity(state, now=now)
    assert state.last_activity_at == now.isoformat()
    assert session_idle_seconds(state, now=now + timedelta(minutes=1)) == 60.0
