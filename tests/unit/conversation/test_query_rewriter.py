from app.services.conversation.models import ConversationState
from app.services.conversation.query_rewriter import (
    QueryRewriter,
    apply_named_product,
    extract_product,
    needs_rewrite,
    rewrite_query,
    should_bind_product,
)
from app.helpers.query_normalize import canonicalize_knowledge_query


def test_simple_query_does_not_rewrite() -> None:
    state = ConversationState(user_message="What is TINY?", product="TINY")
    QueryRewriter().apply(state)
    assert state.query_rewritten == ""
    assert not needs_rewrite(state.user_message, state.product)


def test_contextual_query_rewrites_with_product() -> None:
    rewritten = rewrite_query("What is its battery life?", "TINY")
    assert rewritten == "What is the battery life of TINY?"
    state = ConversationState(user_message="What is its battery life?", product="TINY")
    QueryRewriter().apply(state)
    assert state.query_rewritten == "What is the battery life of TINY?"


def test_greeting_wrapper_becomes_what_is_topic() -> None:
    assert canonicalize_knowledge_query("hi how are u ? i whant to know about CIC") == "What is CIC?"
    assert canonicalize_knowledge_query("tell me about Earkart") == "What is Earkart?"
    state = ConversationState(user_message="hi how are u ? i whant to know about CIC")
    QueryRewriter().apply(state)
    assert state.query_rewritten == "What is CIC?"


def test_filler_prefix_is_stripped_without_inventing_a_topic() -> None:
    state = ConversationState(user_message="ok what is tiny")
    QueryRewriter().apply(state)
    assert state.query_rewritten == "what is tiny"


def test_hearing_loss_clears_stale_product() -> None:
    from app.services.conversation.query_rewriter import apply_named_product

    state = ConversationState(
        user_message="i have 30 percent hearing loss in one ear",
        product="TINY",
    )
    apply_named_product(state)
    assert state.product == ""


def test_hearing_loss_does_not_bind_stale_product() -> None:
    assert not should_bind_product("i have hearing loss", "TINY")
    state = ConversationState(user_message="i have hearing loss", product="TINY")
    QueryRewriter().apply(state)
    assert "TINY" not in (state.query_rewritten or state.user_message)


def test_attribute_query_binds_known_product() -> None:
    state = ConversationState(user_message="what is the price?", product="TINY")
    QueryRewriter().apply(state)
    assert state.query_rewritten == "what is the price of TINY?"


def test_buy_intent_is_not_turned_into_what_is() -> None:
    state = ConversationState(user_message="I want to buy TINY.")
    QueryRewriter().apply(state)
    assert state.query_rewritten == ""


def test_catalog_price_question_does_not_bind_active_product() -> None:
    assert not should_bind_product(
        "give me the price of product that u have in earkart",
        "Bluup",
    )
    state = ConversationState(
        user_message="give me the price of prodcut that u have in earkart",
        product="Bluup",
        query_rewritten="",
        trace={
            "resolved_query": "Give me the price of prodcut Bluup u have in earkart",
            "sub_questions": ["Give me the price of prodcut Bluup u have in earkart"],
        },
    )
    QueryRewriter().apply(state)
    assert "Bluup" not in (state.query_rewritten or state.user_message)


def test_bluup_plus_is_distinct_from_bluup() -> None:
    assert extract_product("what is the price of bluup +") == "Bluup+"
    result = QueryRewriter().normalize(
        ConversationState(user_message="what is the price of bluup +")
    )
    assert "Bluup+" in result.rewritten_query


def test_tell_me_about_topic_strips_conversational_tail() -> None:
    assert canonicalize_knowledge_query("Actually, tell me about BTE first.") == "What is BTE?"
    state = ConversationState(user_message="Actually, tell me about BTE first.", product="TINY")
    QueryRewriter().apply(state)
    assert state.query_rewritten == "What is BTE?"


def test_canonical_query_bypasses_second_pass_when_semantic_used() -> None:
    state = ConversationState(
        user_message="what is tinny warrenty?",
        product="TINY",
        trace={
            "semantic_router_used": True,
            "canonical_query": "What is the warranty of TINY?",
        },
    )
    QueryRewriter().apply(state)
    assert state.query_rewritten == "What is the warranty of TINY?"
    assert state.trace.get("query_rewrite_bypassed") is True


def test_apply_skips_context_rewrite_when_decision_says_not_needed() -> None:
    state = ConversationState(
        user_message="What is the warranty of TINY?",
        product="TINY",
        query_rewritten="",
        trace={"semantic_router_used": True, "needs_rewrite": False, "canonical_query": ""},
    )
    QueryRewriter().apply(state)
    assert state.query_rewritten == ""


def test_apply_falls_back_without_canonical_query() -> None:
    state = ConversationState(
        user_message="What is its battery life?",
        product="TINY",
        trace={"semantic_router_used": True, "canonical_query": ""},
    )
    QueryRewriter().apply(state)
    assert state.query_rewritten == "What is the battery life of TINY?"
    assert state.trace.get("query_rewrite_bypassed") is not True


def test_abte_splits_form_factor() -> None:
    result = QueryRewriter().normalize(ConversationState(user_message="tell me about aBTE"))
    assert "a BTE" in result.rewritten_query


def test_person_alias_rohit_misa() -> None:
    result = QueryRewriter().normalize(ConversationState(user_message="who is rohit misa"))
    assert "Rohit Misra" in result.rewritten_query


def test_yes_after_buy_does_not_rewrite_to_what_is_product() -> None:
    state = ConversationState(
        user_message="yes",
        product="TINY",
        conversation_history=[
            {"role": "user", "content": "i want to buy tiny"},
            {"role": "assistant", "content": "Absolutely! TINY is a great choice."},
        ],
    )
    QueryRewriter().apply(state)
    assert "What is TINY" not in (state.query_rewritten or "")
