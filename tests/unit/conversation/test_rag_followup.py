from tests.unit.conversation.fakes import make_orchestrator
from app.helpers.quick_replies import quick_replies_from_trace


def test_rag_answer_does_not_append_rigid_bullet_followups() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("rag-product", "What is TINY?")
    assert knowledge.queries
    assert "• What is the" not in result.response
    assert "For example, you could ask" not in result.response
    assert "Would you like to know about TINY's warranty, price or battery?" not in result.response
    replies = quick_replies_from_trace(result)
    assert replies
    assert replies[0]["label"] == "I want TINY"


def test_rag_terms_answer_has_no_code_appended_followup_menu() -> None:
    orchestrator, knowledge, *_ = make_orchestrator()
    result = orchestrator.handle("rag-terms", "What are the terms and conditions?")
    assert knowledge.queries
    assert "• What is the return policy?" not in result.response
    assert quick_replies_from_trace(result) == []
