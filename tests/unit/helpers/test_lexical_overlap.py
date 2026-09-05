from app.helpers.lexical_overlap import lexical_coverage_score, tokenize_query_terms


def test_price_query_matches_mrp_heading() -> None:
    score = lexical_coverage_score(
        "what is the price of bluup",
        "tmpulktdi2p > MRP: ₹ 2,900\n5.7.2 Bluup by earKART",
    )
    assert score == 1.0


def test_abte_tokenizes_as_bte() -> None:
    assert "bte" in tokenize_query_terms("tell me about aBTE")


def test_zero_overlap_scores_zero() -> None:
    assert lexical_coverage_score("who is virat khloi", "TINY rechargeable hearing aid") == 0.0
