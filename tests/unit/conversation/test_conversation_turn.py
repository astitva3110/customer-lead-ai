from app.helpers.conversation_reply import greeting_reply
from app.helpers.conversation_turn import is_greeting_only, strip_greeting_prefix


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
