import pytest

from app.kb.url_normalizer import document_identity_key, extract_website, make_document_id, normalize_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://earkart.in/", "https://earkart.in/"),
        ("https://www.Earkart.IN/index.html", "https://earkart.in/"),
        ("https://earkart.in/about-us.html", "https://earkart.in/about-us.html"),
        ("https://earkart.in/about-us.html/", "https://earkart.in/about-us.html"),
        ("https://earkart.com/products/tiny?pr_prod_strat=e5_desc&pr_rec_id=abc", "https://earkart.com/products/tiny"),
        ("https://earkart.in/benefits-t%26c", "https://earkart.in/benefits-t&c"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def test_extract_website() -> None:
    assert extract_website("https://www.earkart.in/about-us.html") == "earkart.in"


def test_document_identity_stable() -> None:
    a = document_identity_key("earkart.in", "https://earkart.in/index.html")
    b = document_identity_key("earkart.in", "https://www.earkart.in/")
    assert a == b


def test_document_id_deterministic() -> None:
    id1 = make_document_id("earkart.in", "https://earkart.in/about-us.html")
    id2 = make_document_id("earkart.in", "https://earkart.in/about-us.html")
    assert id1 == id2
    assert len(id1) == 36
