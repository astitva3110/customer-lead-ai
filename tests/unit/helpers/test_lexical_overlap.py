from app.helpers.lexical_overlap import lexical_coverage_score


def test_price_query_matches_mrp_heading() -> None:
    score = lexical_coverage_score(
        "what is the price of bluup",
        "tmpulktdi2p > MRP: ₹ 2,900\n5.7.2 Bluup by earKART",
    )
    assert score == 1.0
