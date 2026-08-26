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


def test_system_prompt_forbids_value_substitution() -> None:
    assert "Do not substitute a similar value" in GENERATION_SYSTEM_PROMPT
    assert "Do not prepend a company or product overview" in GENERATION_SYSTEM_PROMPT
    assert "Retrieved context is evidence" in GENERATION_SYSTEM_PROMPT
    assert "₹8,900" in GENERATION_SYSTEM_PROMPT
    assert "₹10,999" in GENERATION_SYSTEM_PROMPT
    assert "New York" in GENERATION_SYSTEM_PROMPT
    assert "Noida" in GENERATION_SYSTEM_PROMPT
    assert "grounded=false" in GENERATION_SYSTEM_PROMPT


def test_system_prompt_requires_exact_source_ids() -> None:
    assert "source_ids is mandatory when grounded=true" in GENERATION_SYSTEM_PROMPT
    assert "Never omit source_ids" in GENERATION_SYSTEM_PROMPT
    assert "Do not create, shorten, transform, hash, or" in GENERATION_SYSTEM_PROMPT
    assert "invent source IDs" in GENERATION_SYSTEM_PROMPT
    assert "Never return a source ID that does not appear in the supplied context" in GENERATION_SYSTEM_PROMPT
    assert '"source_ids": ["chunk_about_earkart"]' in GENERATION_SYSTEM_PROMPT
    assert "Section headings are evidence" in GENERATION_SYSTEM_PROMPT


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
    assert "Return ONLY the required JSON object with grounded, answer, and source_ids." in prompt
    assert "Allowed SOURCE_ID values" in prompt
    assert "- chunk_123" in prompt
    assert "Never omit source_ids." in prompt
    assert "source_ids is mandatory when grounded=true." in prompt


def test_user_prompt_can_ask_for_a_greeting_without_new_facts() -> None:
    prompt = build_generation_user_prompt(
        "hi, what is earkart",
        [_hit("chunk_123", "Earkart is a digital-first hearing care platform.")],
        acknowledge_greeting=True,
    )
    assert "brief natural greeting" in prompt
    assert "context only" in prompt
    assert "hi, what is earkart" in prompt
    assert "Do not introduce the company unless the user asked about the company." in prompt
