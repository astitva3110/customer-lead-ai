from app.helpers.conversation_reply import greeting_reply
from app.helpers.conversation_turn import (
    accepting_knowledge_followup,
    is_casual_conversation,
    is_greeting_only,
    strip_greeting_prefix,
    accepts_sales_offer,
)


def test_greeting_only_includes_multilingual_and_stretched_greetings() -> None:
    for message in (
        "namaste",
        "Namaste!",
        "hola",
        "hiiii",
        "heyyy",
        "hellooo",
        "hey there",
        "good evening",
    ):
        assert is_greeting_only(message), message


def test_greeting_reply_matches_user_language() -> None:
    assert "namaste" in greeting_reply("namaste").lower()
    assert "hola" in greeting_reply("hola").lower()
    assert "hey" in greeting_reply("hiiii").lower()


def test_greeting_only_includes_small_talk() -> None:
    for message in (
        "Hi",
        "Hello",
        "Hey",
        "Hey there",
        "Hi, how are you?",
        "Hello, how's it going?",
        "How are you?",
        "Good morning",
    ):
        assert is_greeting_only(message), message


def test_greeting_plus_question_is_not_greeting_only() -> None:
    assert not is_greeting_only("Hi, I want to know about Bluup.")
    assert not is_greeting_only("Hey, what is TINY?")
    assert strip_greeting_prefix("Hi, I want to know about Bluup.") == "I want to know about Bluup."


def test_greeting_reply_is_a_greeting_not_a_company_pitch() -> None:
    text = greeting_reply().lower()
    assert "earkart" not in text
    assert "digital-first" not in text
    assert "help" in text


def test_casual_conversation_covers_thanks_and_pleasantries() -> None:
    for message in (
        "thanks",
        "thank you so much",
        "appreciate it",
        "bye",
        "have a good day",
        "Hi, how are you?",
    ):
        assert is_casual_conversation(message), message
    assert not is_casual_conversation("Who is the CEO of Earkart?")
    assert not is_casual_conversation("What is the return policy?")
    assert not is_casual_conversation("thanks, what is TINY?")


def test_sales_offer_accept_phrases() -> None:
    for message in ("ok", "OK", "yeah", "ok create it", "ok connect me", "yes create it"):
        assert accepts_sales_offer(message), message


def test_accepting_knowledge_followup_requires_info_offer_not_support() -> None:
    followup = "Would you like to know about its battery life?"
    assert accepting_knowledge_followup("yes", followup)
    assert not accepting_knowledge_followup("yes", "Would you like me to open a support ticket?")
    assert accepting_knowledge_followup(
        "yes",
        "Would you like to know about TINY's warranty, price or battery?",
    )
    assert not accepts_sales_offer("what is TINY?")
