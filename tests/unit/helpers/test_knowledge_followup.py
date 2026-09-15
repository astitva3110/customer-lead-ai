from app.helpers.generation_prompt import GENERATION_SYSTEM_PROMPT, build_generation_user_prompt
from app.helpers.knowledge_followup import resolve_followup_message


def test_generation_prompt_asks_for_single_llm_followup() -> None:
    assert "exactly ONE short natural question" in GENERATION_SYSTEM_PROMPT
    assert "not as bullets" in GENERATION_SYSTEM_PROMPT


def test_generation_user_prompt_allows_one_followup_sentence() -> None:
    prompt = build_generation_user_prompt("What is TINY?", hits=[])
    assert "exactly one short follow-up question" in prompt.lower()


def test_legacy_followup_id_still_resolves() -> None:
    assert resolve_followup_message("followup_return") == "What is the return policy?"
