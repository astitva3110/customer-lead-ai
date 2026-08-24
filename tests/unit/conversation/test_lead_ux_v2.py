from app.helpers.conversation_reply import lead_created_reply
from tests.unit.conversation.fakes import make_orchestrator


def _start_lead(orchestrator, cid: str):
    first = orchestrator.handle(cid, "connect me to sales")
    assert first.awaiting_field == "name"
    assert "name" in first.response.lower()
    named = orchestrator.handle(cid, "sudhanshu")
    assert named.user_name == "Sudhanshu"
    assert named.awaiting_field == "phone"
    assert named.lead_collection_active is True
    return named


def test_01_normal_lead_thanks_not_absolutely() -> None:
    orchestrator, *_ = make_orchestrator()
    named = _start_lead(orchestrator, "ux-1")
    assert named.response == "Thanks, Sudhanshu. What's the best number to reach you on?"
    assert "absolutely" not in named.response.lower()


def test_02_knowledge_during_lead_uses_natural_transition() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    _start_lead(orchestrator, "ux-2")
    result = orchestrator.handle("ux-2", "give me details of TINY")
    assert knowledge.queries
    assert "absolutely" not in result.response.lower()
    assert "if you'd like help" in result.response.lower()
    assert "best number to reach you" in result.response.lower()
    assert result.awaiting_field == "phone"
    assert result.lead_collection_active is True
    assert not result.phone


def test_03_invalid_phone_length_stays_on_phone() -> None:
    orchestrator, *_ = make_orchestrator()
    _start_lead(orchestrator, "ux-3")
    result = orchestrator.handle("ux-3", "12345")
    assert not result.phone
    assert result.awaiting_field == "phone"
    assert result.trace["validation_reason"] == "invalid_length"
    assert "10-digit" in result.response.lower()
    assert "which city" not in result.response.lower()


def test_04_invalid_phone_prefix_does_not_move_to_city() -> None:
    orchestrator, *_ = make_orchestrator()
    _start_lead(orchestrator, "ux-4")
    result = orchestrator.handle("ux-4", "1234567890")
    assert not result.phone
    assert result.awaiting_field == "phone"
    assert result.trace["validation_reason"] == "invalid_prefix"
    assert "indian mobile" in result.response.lower()
    assert "which city" not in result.response.lower()


def test_05_valid_phone_advances_to_city() -> None:
    orchestrator, *_rest, lead_tool, _tickets, _store = make_orchestrator()
    _start_lead(orchestrator, "ux-5")
    result = orchestrator.handle("ux-5", "9876543210")
    assert result.phone == "+919876543210"
    assert result.phone_country == "IN"
    assert result.awaiting_field == "city"
    assert result.response == "Thanks! Which city should I note for the team?"
    assert lead_tool.leads == []


def test_06_phone_with_country_code_normalizes() -> None:
    orchestrator, *_ = make_orchestrator()
    _start_lead(orchestrator, "ux-6")
    result = orchestrator.handle("ux-6", "+91 9876543210")
    assert result.phone == "+919876543210"
    assert result.phone_country == "IN"
    assert result.awaiting_field == "city"
    assert result.trace["phone_validation"]["valid"] is True
    assert result.trace["phone_validation"]["normalized_input"] == "9876543210"


def test_07_complete_lead_creates_and_thanks() -> None:
    orchestrator, *_rest, lead_tool, _tickets, _store = make_orchestrator()
    _start_lead(orchestrator, "ux-7")
    orchestrator.handle("ux-7", "9876543210", country="IN")
    result = orchestrator.handle("ux-7", "Delhi")
    assert result.city == "Delhi"
    assert result.phone == "+919876543210"
    assert lead_tool.leads
    assert lead_tool.leads[0].name == "Sudhanshu"
    assert lead_tool.leads[0].phone == "+919876543210"
    assert result.response == lead_created_reply("Sudhanshu")


def test_knowledge_without_active_lead_does_not_ask_phone() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("ux-knowledge", "give me details of TINY")
    assert knowledge.queries
    assert result.lead_collection_active is False
    assert "best number" not in result.response.lower()
    assert result.awaiting_field == ""
