from app.helpers.bot_guidance import (
    CAPABILITY_REPLY,
    UNCLEAR_REDIRECT_REPLY,
    looks_like_capability_question,
    looks_like_unclear_user_message,
)
from app.helpers.hearing_symptom_normalize import analyze_hearing_symptom
from app.services.conversation.models import ChatMode, ConversationState, TurnIntent
from app.services.conversation.router import ChatRouter
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import make_graph_stack


def test_capability_questions() -> None:
    for message in (
        "What can you do?",
        "How can you help?",
        "Kya karooo",
        "Kya karooo ho",
        "Kya karooo n",
        "Bolo",
        "Sir",
    ):
        assert looks_like_capability_question(message), message


def test_product_questions_are_not_capability() -> None:
    assert not looks_like_capability_question("What can you tell me about TINY?")
    assert not looks_like_capability_question("What is TINY?")
    assert not looks_like_capability_question("what can I try?")


def test_unclear_prompts() -> None:
    for message in (
        "AAP KAHAN HO",
        "Mai aaaauuu",
        "Traaas hore",
        "Where are you",
    ):
        assert looks_like_unclear_user_message(message), message


def test_sunai_problem_is_hearing_not_unclear() -> None:
    for message in ("Sunai problem hore", "1aaankh sunai problem hore"):
        assert not looks_like_unclear_user_message(message), message
        assert analyze_hearing_symptom(message).matched


def test_router_keeps_capability_off_rag() -> None:
    routed = ChatRouter().route(ConversationState(user_message="Kya karooo"))
    assert routed.current_turn_intent == TurnIntent.GENERAL
    assert routed.trace.get("should_retrieve") is False


def test_orchestrator_capability_and_unclear_do_not_hit_kb() -> None:
    knowledge = FakeKnowledge()
    orchestrator, *_ = make_graph_stack(knowledge=knowledge)
    cid = "guide-1"
    first = orchestrator.handle(cid, "Hi")
    assert "hearing" in first.response.lower() or "help" in first.response.lower()
    asked = orchestrator.handle(cid, "Kya karooo")
    assert knowledge.queries == []
    assert asked.response == CAPABILITY_REPLY
    unclear = orchestrator.handle(cid, "AAP KAHAN HO")
    assert unclear.response == UNCLEAR_REDIRECT_REPLY
    assert "knowledge base" not in unclear.response.lower()


def test_kb_gap_redirects_to_earkart_topics() -> None:
    knowledge = FakeKnowledge(chunks=[])
    orchestrator, *_ = make_graph_stack(knowledge=knowledge)
    result = orchestrator.handle("guide-gap", "Who founded Earkart?")
    assert result.response == INSUFFICIENT_INFORMATION_MESSAGE
    assert "earkart" in result.response.lower()
    assert "hearing" in result.response.lower()
    assert result.mode == ChatMode.KNOWLEDGE
