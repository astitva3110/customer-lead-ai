"""Tests for Phase 11.8 targeted chunk inspection."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.retrieval_diagnostic.competitor_classifier import (
    classify_competitor,
    targeted_diagnosis,
)
from app.kb.evaluation.retrieval_diagnostic.targeted_inspection import (
    _best_expected_chunk_in_hits,
    _format_result,
)


def test_expected_chunk_rank_in_top_results() -> None:
    hits = [
        {"rank": 1, "chunk_id": "noise", "similarity": 0.9},
        {"rank": 5, "chunk_id": "answer", "similarity": 0.8},
    ]
    best_id, rank, sim = _best_expected_chunk_in_hits(hits, ["answer", "missing"])
    assert best_id == "answer"
    assert rank == 5
    assert sim == 0.8


def test_document_retrieved_chunk_missed_diagnosis() -> None:
    assert (
        targeted_diagnosis(
            expected_chunk_rank=None,
            expected_document_best_rank=2,
            expected_chunk_found=True,
        )
        == "DOCUMENT_RETRIEVED_CHUNK_MISSED"
    )


def test_document_and_chunk_retrieved_diagnosis() -> None:
    assert (
        targeted_diagnosis(
            expected_chunk_rank=3,
            expected_document_best_rank=3,
            expected_chunk_found=True,
        )
        == "DOCUMENT_AND_CHUNK_RETRIEVED"
    )


def test_document_not_retrieved_diagnosis() -> None:
    assert (
        targeted_diagnosis(
            expected_chunk_rank=None,
            expected_document_best_rank=None,
            expected_chunk_found=True,
        )
        == "DOCUMENT_NOT_RETRIEVED"
    )


def test_competitor_classification_image() -> None:
    hit: dict[str, Any] = {
        "chunk_id": "x",
        "text": "Bluup > Bluup\n\n![](https://earkart.com/icon.png)",
        "section_path": ["Bluup"],
        "token_count": 8,
    }
    assert classify_competitor(hit, {"answer"}) == "IMAGE_OR_ICON"


def test_competitor_classification_cross_sell() -> None:
    hit = {
        "chunk_id": "x",
        "text": "You may also like Bluup Regular price 999",
        "section_path": ["TINY", "You may also like"],
        "token_count": 20,
    }
    assert classify_competitor(hit, set()) == "PRODUCT_CROSS_SELL"


def test_format_result_truncates_after_rank_10() -> None:
    long_text = "word " * 200
    item = {"chunk_id": "c1", "content": long_text, "similarity": 0.5}
    full = _format_result(5, item, truncate_after=None)
    short = _format_result(15, item, truncate_after=500)
    assert len(full["text"]) > 500
    assert len(short["text"]) <= 500
