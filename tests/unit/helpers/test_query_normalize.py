from app.helpers.query_normalize import (
    looks_like_informational_question,
    looks_like_knowledge_request,
    looks_like_price_query,
)


def test_knowledge_request_includes_person_policy_and_company_facts() -> None:
    for message in (
        "Who is the CEO of Earkart?",
        "Who founded Earkart?",
        "What is the return policy?",
        "Where is Earkart based?",
        "How much does TINY cost?",
    ):
        assert looks_like_knowledge_request(message), message
        assert looks_like_informational_question(message), message


def test_informational_questions_include_yes_no_product_facts() -> None:
    assert looks_like_informational_question("Is TINY waterproof?")
    assert looks_like_informational_question("Does Earkart ship internationally?")
    assert not looks_like_informational_question("thanks")
    assert not looks_like_informational_question("Hi, how are you?")


def test_price_query_detection() -> None:
    assert looks_like_price_query("What is the price of Bluup?")
    assert looks_like_price_query("How much does TINY cost?")
    assert not looks_like_price_query("What is TINY?")
    assert not looks_like_price_query("What is the warranty?")
