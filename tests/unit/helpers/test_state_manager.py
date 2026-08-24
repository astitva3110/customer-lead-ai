from __future__ import annotations

from app.helpers.state_manager import parse_state_manager_result


def test_parse_state_manager_result_accepts_schema() -> None:
    raw = """
    {
      "conversation_status": "SALE",
      "current_status": "KNOWLEDGE",
      "active_product": "TINY",
      "diverge": false,
      "sub_questions": ["What are the features of TINY?"],
      "sales_interest": true,
      "lead": {
        "status": "NOT_STARTED",
        "collected_fields": {},
        "missing_fields": []
      },
      "next_action": "ANSWER_KNOWLEDGE"
    }
    """
    parsed = parse_state_manager_result(raw)
    assert parsed is not None
    assert parsed.conversation_status == "SALE"
    assert parsed.current_status == "KNOWLEDGE"
    assert parsed.active_product == "TINY"
    assert parsed.diverge is False
    assert parsed.sub_questions == ["What are the features of TINY?"]
    assert parsed.next_action == "ANSWER_KNOWLEDGE"
    assert parsed.lead.status == "NOT_STARTED"


def test_parse_resume_lead_after_knowledge() -> None:
    raw = """
    {
      "conversation_status": "SALE",
      "current_status": "KNOWLEDGE",
      "active_product": "TINY",
      "diverge": false,
      "sub_questions": ["What is the warranty of TINY?"],
      "sales_interest": true,
      "lead": {
        "status": "IN_PROGRESS",
        "collected_fields": {"name": "Rahul"},
        "missing_fields": ["phone", "city"]
      },
      "next_action": "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD"
    }
    """
    parsed = parse_state_manager_result(raw)
    assert parsed is not None
    assert parsed.next_action == "ANSWER_KNOWLEDGE_THEN_RESUME_LEAD"
    assert parsed.lead.collected_fields["name"] == "Rahul"
    assert parsed.lead.missing_fields == ["phone", "city"]
