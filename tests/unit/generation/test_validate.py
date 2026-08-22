from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.helpers.generation_validate import explain_generation_payload, unique_preserve_order, validate_generation_payload


ALLOWED = {"chunk_1", "chunk_a", "chunk_b"}


def test_grounded_answer_accepts_known_source() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "Earkart provides...", "source_ids": ["chunk_1"]},
        ALLOWED,
    )
    assert result is not None
    assert result.grounded is True
    assert result.answer == "Earkart provides..."
    assert result.source_ids == ["chunk_1"]


def test_ungrounded_uses_safe_message() -> None:
    result = validate_generation_payload(
        {"grounded": False, "answer": "something else", "source_ids": []},
        ALLOWED,
    )
    assert result is not None
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.source_ids == []


def test_unknown_source_id_fails_closed() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "...", "source_ids": ["does-not-exist"]},
        ALLOWED,
    )
    assert result is None


def test_unknown_source_among_valid_fails_closed() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "...", "source_ids": ["chunk_a", "fake_chunk"]},
        ALLOWED,
    )
    assert result is None


def test_empty_source_ids_with_grounded_true_fails_closed() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "Earkart is a company.", "source_ids": []},
        ALLOWED,
    )
    assert result is None


def test_duplicate_source_ids_are_deduped() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "ok", "source_ids": ["chunk_1", "chunk_1"]},
        ALLOWED,
    )
    assert result is not None
    assert result.source_ids == ["chunk_1"]


def test_source_id_from_other_document_fails_closed() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "ok", "source_ids": ["other-doc-chunk"]},
        ALLOWED,
    )
    assert result is None


def test_grounded_yes_string_is_invalid() -> None:
    result = validate_generation_payload(
        {"grounded": "yes", "answer": "ok", "source_ids": ["chunk_1"]},
        ALLOWED,
    )
    assert result is None


def test_missing_source_ids_is_invalid() -> None:
    result = validate_generation_payload({"grounded": True, "answer": "ok"}, ALLOWED)
    assert result is None
    explained = explain_generation_payload({"grounded": True, "answer": "ok"}, ALLOWED)
    assert explained["validator_reason"] == "missing_fields"
    assert explained["validator_result"] is False


def test_unknown_source_id_explain_reason() -> None:
    payload = {"grounded": True, "answer": "...", "source_ids": ["not-in-context"]}
    assert validate_generation_payload(payload, ALLOWED) is None
    explained = explain_generation_payload(payload, ALLOWED)
    assert explained["validator_reason"] == "source_id_not_found"
    assert explained["invalid_source_ids"] == ["not-in-context"]
    assert explained["validator_result"] is False


def test_null_source_ids_is_invalid() -> None:
    result = validate_generation_payload(
        {"grounded": True, "answer": "ok", "source_ids": None},
        ALLOWED,
    )
    assert result is None


def test_extra_fields_are_ignored() -> None:
    result = validate_generation_payload(
        {
            "grounded": True,
            "answer": "ok",
            "source_ids": ["chunk_1"],
            "confidence": 0.9,
        },
        ALLOWED,
    )
    assert result is not None
    assert result.source_ids == ["chunk_1"]


def test_unique_preserve_order() -> None:
    assert unique_preserve_order(["b", "a", "b", "c"]) == ["b", "a", "c"]
