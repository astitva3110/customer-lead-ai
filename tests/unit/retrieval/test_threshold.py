from dataclasses import replace

from app.services.retrieval.models import RetrievalCandidate
from app.services.retrieval.threshold import apply_rerank_threshold, take_final_k


def _ranked(chunk_id: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=chunk_id,
        rerank_score=score,
    )


def test_threshold_drops_candidates_below_min_score() -> None:
    kept = apply_rerank_threshold(
        [_ranked("A", 0.91), _ranked("B", 0.82), _ranked("C", 0.40)],
        0.80,
    )
    assert [item.chunk_id for item in kept] == ["A", "B"]


def test_threshold_rejects_all_when_none_meet_min_score() -> None:
    kept = apply_rerank_threshold(
        [_ranked("A", 0.50), _ranked("B", 0.40)],
        0.80,
    )
    assert kept == []


def test_take_final_k() -> None:
    candidates = [_ranked("A", 0.9), _ranked("B", 0.8), _ranked("C", 0.7)]
    assert [item.chunk_id for item in take_final_k(candidates, 2)] == ["A", "B"]


def test_threshold_skips_missing_rerank_score() -> None:
    missing = replace(_ranked("Z", 0.9), rerank_score=None)
    kept = apply_rerank_threshold([missing, _ranked("A", 0.91)], 0.80)
    assert [item.chunk_id for item in kept] == ["A"]
