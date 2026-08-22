from app.services.conversation.models import ChatMode, LeadStatus, TicketStatus
from app.helpers.conversation_reply import lead_created_reply, ticket_created_reply
from tests.unit.conversation.fakes import make_orchestrator


def test_knowledge_to_lead_preserves_product() -> None:
    orchestrator, knowledge, _llm, lead_tool, *_ = make_orchestrator()
    first = orchestrator.handle("switch-1", "What is Radius M16?")
    assert first.mode == ChatMode.KNOWLEDGE
    assert first.product == "Radius M16"
    second = orchestrator.handle("switch-1", "I want to buy it.")
    assert second.mode == ChatMode.LEAD
    assert second.product == "Radius M16"
    assert second.lead_intent is True
    assert knowledge.queries == ["What is Radius M16?"]
    assert lead_tool.leads == []


def test_lead_to_knowledge_then_returns_to_lead() -> None:
    orchestrator, *_ = make_orchestrator()
    orchestrator.handle("switch-2", "I want to buy Radius M16.")
    result = orchestrator.handle("switch-2", "What is the warranty?")
    assert result.mode == ChatMode.LEAD
    assert result.lead_status == LeadStatus.COLLECTING
    assert result.product == "Radius M16"
    assert "warranty" in result.response.lower() or result.response
    assert "please provide your phone number first" not in result.response.lower()


def test_support_to_knowledge_preserves_support_state() -> None:
    orchestrator, *_ = make_orchestrator()
    first = orchestrator.handle("switch-3", "My hearing aid isn't working.")
    assert first.mode == ChatMode.SUPPORT
    assert first.support_issue
    result = orchestrator.handle("switch-3", "Actually, how long is the warranty?")
    assert result.mode == ChatMode.SUPPORT
    assert result.ticket_status == TicketStatus.COLLECTING
    assert result.support_issue
    assert "please provide" not in result.response.lower()


def test_knowledge_to_support() -> None:
    orchestrator, *_ = make_orchestrator()
    orchestrator.handle("switch-4", "What is Radius M16?")
    result = orchestrator.handle("switch-4", "My Radius M16 stopped working.")
    assert result.mode == ChatMode.SUPPORT
    assert result.product == "Radius M16"


def test_lead_field_collection_does_not_call_knowledge_or_qwen() -> None:
    orchestrator, knowledge, _llm, lead_tool, *_ = make_orchestrator()
    cid = "lead-flow"
    orchestrator.handle(cid, "I want to buy Radius M16.")
    orchestrator.handle(cid, "Please call me.")
    orchestrator.handle(cid, "+91 9876543210")
    orchestrator.handle(cid, "Ada")
    result = orchestrator.handle(cid, "Noida")
    assert knowledge.queries == []
    assert result.lead_status == LeadStatus.CREATED
    assert result.response == lead_created_reply("Ada")
    assert lead_tool.leads[0].city == "Noida"


def test_support_ticket_creation_does_not_call_qwen() -> None:
    orchestrator, knowledge, _llm, _, ticket_tool, _ = make_orchestrator()
    cid = "support-flow"
    orchestrator.handle(cid, "My hearing aid isn't working.")
    orchestrator.handle(cid, "Please create a support ticket.")
    orchestrator.handle(cid, "Ada")
    orchestrator.handle(cid, "Radius M16")
    result = orchestrator.handle(cid, "+91 9876543210")
    assert knowledge.queries == []
    assert result.ticket_status == TicketStatus.CREATED
    assert result.response == ticket_created_reply("Ada")
    assert ticket_tool.tickets
