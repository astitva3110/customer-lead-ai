from app.helpers.user_language import (
    detect_user_language,
    language_instruction,
    localized_text,
    response_language,
    touch_response_language,
)
from app.services.conversation.models import ConversationState


def test_detect_english() -> None:
    assert detect_user_language("What is the price of TINY?") == "en"


def test_detect_hindi_devanagari() -> None:
    assert detect_user_language("मुझे हियरिंग एड चाहिए") == "hi"


def test_detect_hinglish() -> None:
    assert detect_user_language("mujhe hearing aid chahiye") == "hinglish"


def test_detect_mixed_script_is_hinglish() -> None:
    assert detect_user_language("TINY ka price kya hai?") == "hinglish"


def test_short_ack_does_not_change_language() -> None:
    assert detect_user_language("yes") is None
    assert detect_user_language("haan") is None


def test_touch_response_language_persists_in_user_context() -> None:
    state = ConversationState(user_message="mujhe hearing aid chahiye")
    touch_response_language(state)
    assert state.user_context["response_language"] == "hinglish"
    assert state.trace["response_language"] == "hinglish"


def test_response_language_reuses_session_for_short_reply() -> None:
    state = ConversationState(
        user_message="yes",
        user_context={"response_language": "hi"},
    )
    touch_response_language(state)
    assert response_language(state) == "hi"


def test_language_instruction_variants() -> None:
    assert "English" in language_instruction("en")
    assert "Devanagari" in language_instruction("hi")
    assert "Hinglish" in language_instruction("hinglish")


def test_localized_lead_created_english() -> None:
    text = localized_text("lead_created", "en", name=", Rahul")
    assert "Thanks, Rahul" in text
    assert "our team" in text


def test_localized_lead_created_hindi() -> None:
    text = localized_text("lead_created", "hi", name=", Rahul")
    assert "धन्यवाद" in text


def test_build_conversation_prompt_includes_language_instruction() -> None:
    from app.helpers.conversation_prompt import build_conversation_user_prompt

    state = ConversationState(user_message="mujhe hearing aid chahiye")
    touch_response_language(state)
    prompt = build_conversation_user_prompt(state)
    assert "hinglish" in prompt.lower()
    assert "Hinglish" in prompt


def test_localized_capability_reply_hinglish() -> None:
    text = localized_text("capability_reply", "hinglish")
    assert "hearing aids" in text.lower()
    assert "help" in text.lower()


def test_build_generation_prompt_includes_language_instruction() -> None:
    from app.helpers.generation_prompt import build_generation_user_prompt

    class Hit:
        chunk_id = "c1"
        document_title = "T"
        section_path = []
        document_id = "d1"
        text = "TINY is a hearing aid."

    prompt = build_generation_user_prompt(
        "mujhe TINY ke baare mein batao",
        [Hit()],
        response_language="hinglish",
    )
    assert "Hinglish" in prompt
    assert "TINY kya hai?" in prompt


def test_build_generation_prompt_includes_hindi_example() -> None:
    from app.helpers.generation_prompt import build_generation_user_prompt

    class Hit:
        chunk_id = "c1"
        document_title = "T"
        section_path = []
        document_id = "d1"
        text = "TINY is a hearing aid."

    prompt = build_generation_user_prompt(
        "TINY kya hai?",
        [Hit()],
        response_language="hi",
    )
    assert "Devanagari" in prompt
    assert "रिचार्जेबल" in prompt
