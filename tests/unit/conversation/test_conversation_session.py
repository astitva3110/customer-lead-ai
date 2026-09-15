from datetime import datetime, timedelta, timezone

from app.services.conversation.models import ConversationGoal, ConversationState, LeadWorkflow
from tests.unit.conversation.fakes import make_orchestrator


def test_orchestrator_resets_whatsapp_state_after_idle_timeout() -> None:
    orchestrator, *_ = make_orchestrator()
    cid = "whatsapp:+919876543210"
    stale = ConversationState(
        conversation_id=cid,
        channel="whatsapp",
        origin="whatsapp",
        phone="+919876543210",
        conversation_history=[{"role": "user", "content": "I want TINY"}],
        product="TINY",
        conversation_goal=ConversationGoal.LEAD,
        lead_workflow=LeadWorkflow.DISCUSSING_PRODUCT,
        last_activity_at=(datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(),
    )
    orchestrator._store.save(stale)

    result = orchestrator.handle(
        cid,
        "Hi",
        channel="whatsapp",
        origin="whatsapp",
        phone="9876543210",
    )

    assert result.conversation_goal == ConversationGoal.NONE
    assert result.lead_workflow == LeadWorkflow.NONE
    assert result.product == ""
    assert all("TINY" not in item.get("content", "") for item in result.conversation_history)
