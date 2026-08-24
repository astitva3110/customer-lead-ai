from app.helpers.phone import (
    looks_like_phone_attempt,
    parse_country_reply,
    parse_phone,
    validate_phone_number,
)


def test_valid_ten_digit_indian_mobile() -> None:
    result = validate_phone_number("9876543210")
    assert result.valid is True
    assert result.normalized_value == "9876543210"
    assert result.e164 == "+919876543210"
    assert result.country_code == "+91"
    assert result.national_number == "9876543210"
    assert parse_phone("9876543210") == ("+919876543210", "IN")


def test_normalizes_country_code_and_leading_zero() -> None:
    for raw in ("+91 9876543210", "+919876543210", "09876543210", "919876543210"):
        result = validate_phone_number(raw)
        assert result.valid is True
        assert result.normalized_value == "9876543210"
        assert result.e164 == "+919876543210"


def test_invalid_length() -> None:
    result = validate_phone_number("12345")
    assert result.valid is False
    assert result.reason == "invalid_length"
    assert parse_phone("12345") is None


def test_invalid_prefix() -> None:
    result = validate_phone_number("1234567890")
    assert result.valid is False
    assert result.reason == "invalid_prefix"
    assert parse_phone("1234567890") is None


def test_does_not_treat_knowledge_question_as_phone() -> None:
    result = validate_phone_number("give me details of TINY")
    assert result.valid is False
    assert result.reason == "not_found"
    assert looks_like_phone_attempt("give me details of TINY") is False
    assert looks_like_phone_attempt("12345") is True


def test_parse_country_reply() -> None:
    assert parse_country_reply("India") == "IN"
    assert parse_country_reply("US") == "US"
