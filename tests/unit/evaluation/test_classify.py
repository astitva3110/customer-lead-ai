from __future__ import annotations

from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.evaluation.catalog import items_by_key
from app.evaluation.classify import classify_conversation, classify_turn
from app.evaluation.schema import ConversationExpected, TurnMessage
from tests.unit.evaluation.helpers import knowledge_trace, make_conversation


def test_expectation_parsing():
    conversation = make_conversation()
    assert conversation.expected.rag_turns == [1, 2]
    assert conversation.messages[1].needs_rewrite is True
    assert conversation.expected.knowledge_keys[0] == "product:tiny"


def test_trace_ingestion_and_pass(catalog):
    conversation = make_conversation()
    mapping = items_by_key(catalog)
    first = classify_turn(conversation, conversation.messages[0], knowledge_trace(), catalog=mapping, prior_tools=[])
    assert first.passed
    assert first.failure_type == "NONE"


def test_guardrail_false_positive(catalog):
    conversation = make_conversation()
    mapping = items_by_key(catalog)
    result = classify_turn(
        conversation,
        conversation.messages[0],
        knowledge_trace(blocked=True),
        catalog=mapping,
        prior_tools=[],
    )
    assert result.failure_type == "GUARDRAIL_FAILURE"
    assert result.first_failing_layer == "guardrail"


def test_retrieval_failure_not_generation(catalog):
    conversation = make_conversation()
    mapping = items_by_key(catalog)
    trace = knowledge_trace(
        raw=[{"rank": 1, "chunk_id": "other", "title": "other", "section_path": "unrelated", "text_preview": "office hours"}],
        final=[{"chunk_id": "other", "section_path": ["unrelated"], "text": "office hours"}],
        grounded=False,
        reason="empty_hits",
        response=INSUFFICIENT_INFORMATION_MESSAGE,
    )
    result = classify_turn(conversation, conversation.messages[0], trace, catalog=mapping, prior_tools=[])
    assert result.failure_type == "RETRIEVAL_FAILURE"
    assert result.first_failing_layer == "retrieval"


def test_reranker_failure_only_when_raw_had_evidence(catalog):
    conversation = make_conversation()
    mapping = items_by_key(catalog)
    raw = [{"rank": 1, "chunk_id": "c-tiny", "title": "TINY", "section_path": "6.1 TINY", "text_preview": "TINY features: rechargeable 16-channel"}]
    dropped = [{"rank": 1, "chunk_id": "other", "title": "other", "section_path": "unrelated", "text_preview": "office"}]
    trace = knowledge_trace(raw=raw, reranked=dropped, final=dropped, grounded=False, response=INSUFFICIENT_INFORMATION_MESSAGE)
    result = classify_turn(conversation, conversation.messages[0], trace, catalog=mapping, prior_tools=[])
    assert result.failure_type == "RERANKER_FAILURE"


def test_corpus_gap_pass_vs_grounding(catalog):
    conversation = make_conversation(
        category="CORPUS_GAP",
        messages=[
            TurnMessage(turn=1, text="What is OMNI?", needs_rag=True, answerable=False, knowledge_keys=["product:omni"]),
        ],
        expected=ConversationExpected(answerable=False, knowledge_keys=["product:omni"], goal="KNOWLEDGE", rag_turns=[1], corpus_gap_turns=[1]),
    )
    mapping = items_by_key(catalog)
    ok = classify_turn(
        conversation,
        conversation.messages[0],
        knowledge_trace(raw=[], final=[], grounded=False, reason="llm_ungrounded", response=INSUFFICIENT_INFORMATION_MESSAGE),
        catalog=mapping,
        prior_tools=[],
    )
    assert ok.failure_type == "CORPUS_GAP"
    assert ok.passed
    hallucinated = classify_turn(
        conversation,
        conversation.messages[0],
        knowledge_trace(grounded=True, reason="ok", response="Signia lasts 40 hours."),
        catalog=mapping,
        prior_tools=[],
    )
    assert hallucinated.failure_type == "GROUNDING_FAILURE"
    assert hallucinated.passed is False


def test_product_switch_state_failure(catalog):
    conversation = make_conversation(
        category="LEAD_PRODUCT_SWITCH",
        messages=[
            TurnMessage(turn=1, text="Actually I mean Bluup.", expected_intent="CONTEXT_UPDATE", expected_product="Bluup", explicit_product_correction=True),
        ],
        expected=ConversationExpected(answerable=True, goal="LEAD", expected_product="Bluup", rag_turns=[]),
    )
    mapping = items_by_key(catalog)
    trace = knowledge_trace(intent="CONTEXT_UPDATE", needs_rag=False, product="TINY", goal="LEAD")
    result = classify_turn(conversation, conversation.messages[0], trace, catalog=mapping, prior_tools=[])
    assert result.failure_type == "STATE_FAILURE"
    assert result.pattern == "STATE_FAILURE_PRODUCT_SWITCH"


def test_first_failure_wins_for_conversation(catalog):
    conversation = make_conversation()
    mapping = items_by_key(catalog)
    first = classify_turn(
        conversation,
        conversation.messages[0],
        knowledge_trace(raw=[{"rank": 1, "chunk_id": "x", "title": "x", "section_path": "nope", "text_preview": "nope"}]),
        catalog=mapping,
        prior_tools=[],
    )
    later = classify_turn(
        conversation,
        conversation.messages[1],
        knowledge_trace(rewrite=True, reason="invalid_json"),
        catalog=mapping,
        prior_tools=[],
    )
    overall = classify_conversation(conversation, [first, later], {})
    assert overall.failure_type == "RETRIEVAL_FAILURE"


def test_lead_hypothesis_is_not_a_candidate_failure(catalog):
    conversation = make_conversation(
        category="LEAD",
        messages=[
            TurnMessage(turn=1, text="I want to buy TINY.", expected_intent="LEAD_INTENT", expected_product="TINY"),
        ],
        expected=ConversationExpected(
            answerable=True,
            goal="LEAD",
            rag_turns=[],
            expected_lead_created=True,
            expected_product="TINY",
        ),
    )
    mapping = items_by_key(catalog)
    result = classify_turn(
        conversation,
        conversation.messages[0],
        knowledge_trace(intent="LEAD_INTENT", needs_rag=False, goal="LEAD"),
        catalog=mapping,
        prior_tools=[],
    )
    assert result.passed
    assert result.candidate is False
    assert result.failure_type == "NONE"
    overall = classify_conversation(conversation, [result], {})
    assert overall.candidate is False


def test_unverified_knowledge_key_is_observation_only(catalog):
    conversation = make_conversation(
        messages=[
            TurnMessage(turn=1, text="What is Signia?", needs_rag=True, answerable=True, knowledge_keys=["product:not_in_catalog"]),
        ],
        expected=ConversationExpected(answerable=True, knowledge_keys=["product:not_in_catalog"], goal="KNOWLEDGE", rag_turns=[1]),
    )
    mapping = items_by_key(catalog)
    result = classify_turn(
        conversation,
        conversation.messages[0],
        knowledge_trace(raw=[], final=[], grounded=False, reason="empty_hits", response="I don't know."),
        catalog=mapping,
        prior_tools=[],
    )
    assert result.passed
    assert result.failure_type == "NONE"
    assert result.knowledge_keys == []
