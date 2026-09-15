from app.helpers.generation_prompt import GENERATION_SYSTEM_PROMPT, build_generation_user_prompt
from app.kb.evaluation.models import RankedHit


def _hit(chunk_id: str, text: str) -> RankedHit:
    return RankedHit(
        rank=1,
        similarity=0.9,
        chunk_id=chunk_id,
        document_id="doc-1",
        section_path=["About"],
        token_count=8,
        content_type="paragraph",
        text=text,
        page_number=1,
        document_title="Pricelist",
    )


def test_system_prompt_asks_for_answer_json() -> None:
    assert '{"answer": "..."}' in GENERATION_SYSTEM_PROMPT
    assert "Do not prepend a company or product overview" in GENERATION_SYSTEM_PROMPT
    assert "grounded=false" not in GENERATION_SYSTEM_PROMPT
    assert "source_ids is mandatory" not in GENERATION_SYSTEM_PROMPT


def test_source_block_puts_section_heading_in_content() -> None:
    hit = RankedHit(
        rank=1,
        similarity=0.9,
        chunk_id="chunk_bluup",
        document_id="doc-1",
        section_path=["tmpulktdi2p", "MRP: ₹ 2,900"],
        token_count=8,
        content_type="paragraph",
        text="5.7.2 Bluup by earKART\nNoise Reduction: -28 dB",
        page_number=1,
        document_title="Pricelist",
    )
    prompt = build_generation_user_prompt("What is the price of Bluup?", [hit])
    assert "MRP: ₹ 2,900" in prompt
    assert prompt.index("Content:") < prompt.index("5.7.2 Bluup by earKART")


def test_user_prompt_uses_stable_source_ids() -> None:
    prompt = build_generation_user_prompt(
        "What is Earkart?",
        [_hit("chunk_123", "Earkart is a digital-first hearing care platform.")],
    )
    assert "[SOURCE_ID: chunk_123]" in prompt
    assert "Earkart is a digital-first hearing care platform." in prompt
    assert "What is Earkart?" in prompt
    assert "Return ONLY the required JSON object with answer." in prompt
    assert "Never omit source_ids." not in prompt


def test_user_prompt_includes_hearing_health_guidance() -> None:
    prompt = build_generation_user_prompt(
        "i have 30 percent hearing loss in one ear",
        [_hit("chunk_123", "An audiologist can evaluate your hearing.")],
    )
    assert "hearing-health or hearing-loss question" in prompt
    assert "Do not ask about product battery" in prompt
    assert "Do not end with a follow-up question." in prompt


def test_user_prompt_includes_response_language_instruction() -> None:
    prompt = build_generation_user_prompt(
        "kya hai TINY",
        [_hit("chunk_123", "TINY is a rechargeable hearing aid.")],
        response_language="hi",
    )
    assert "Reply entirely in Hindi using Devanagari script." in prompt


def test_user_prompt_can_ask_for_a_greeting_without_new_facts() -> None:
    prompt = build_generation_user_prompt(
        "hi, what is earkart",
        [_hit("chunk_123", "Earkart is a digital-first hearing care platform.")],
        acknowledge_greeting=True,
    )
    assert "brief natural greeting" in prompt
    assert "hi, what is earkart" in prompt
    assert "Do not introduce the company unless the user asked about the company." in prompt
