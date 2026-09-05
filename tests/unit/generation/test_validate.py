from app.helpers.generation_validate import explain_generation_payload, unique_preserve_order, validate_generation_payload


ALLOWED = {"chunk_1", "chunk_a", "chunk_b"}


def test_answer_only_payload_is_accepted() -> None:
    result = validate_generation_payload({"answer": "Earkart provides..."}, ALLOWED)
    assert result is not None
    assert result.grounded is True
    assert result.answer == "Earkart provides..."
    assert result.source_ids == ["chunk_1", "chunk_a", "chunk_b"]


def test_known_source_ids_are_kept() -> None:
    result = validate_generation_payload(
        {"answer": "Earkart provides...", "source_ids": ["chunk_1"]},
        ALLOWED,
    )
    assert result is not None
    assert result.source_ids == ["chunk_1"]


def test_unknown_source_id_still_returns_answer() -> None:
    result = validate_generation_payload(
        {"answer": "The MRP is 8900.", "source_ids": ["does-not-exist"]},
        ALLOWED,
    )
    assert result is not None
    assert result.answer == "The MRP is 8900."
    assert result.source_ids == ["chunk_1", "chunk_a", "chunk_b"]


def test_duplicate_source_ids_are_deduped() -> None:
    result = validate_generation_payload(
        {"answer": "ok", "source_ids": ["chunk_1", "chunk_1"]},
        ALLOWED,
    )
    assert result is not None
    assert result.source_ids == ["chunk_1"]


def test_missing_answer_is_invalid() -> None:
    result = validate_generation_payload({"source_ids": ["chunk_1"]}, ALLOWED)
    assert result is None
    explained = explain_generation_payload({"source_ids": ["chunk_1"]}, ALLOWED)
    assert explained["validator_reason"] == "invalid_answer"
    assert explained["validator_result"] is False


def test_unknown_source_id_explain_keeps_answer_ok() -> None:
    payload = {"answer": "...", "source_ids": ["not-in-context"]}
    assert validate_generation_payload(payload, ALLOWED) is not None
    explained = explain_generation_payload(payload, ALLOWED)
    assert explained["validator_reason"] == "ok"
    assert explained["invalid_source_ids"] == ["not-in-context"]
    assert explained["validator_result"] is True


def test_extra_fields_are_ignored() -> None:
    result = validate_generation_payload(
        {"answer": "ok", "source_ids": ["chunk_1"], "confidence": 0.9, "grounded": True},
        ALLOWED,
    )
    assert result is not None
    assert result.source_ids == ["chunk_1"]


def test_unique_preserve_order() -> None:
    assert unique_preserve_order(["b", "a", "b", "c"]) == ["b", "a", "c"]
