from app.helpers.keyword_query import fts_or_query


def test_fts_or_query_unions_tokens() -> None:
    assert fts_or_query("Radius 16 MRP 27500") == "radius | 16 | mrp | 27500"


def test_fts_or_query_empty() -> None:
    assert fts_or_query("   ") is None
    assert fts_or_query("???") is None
