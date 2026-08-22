"""Phase 15 root-cause classification for KB V2 retrieval failures."""

from __future__ import annotations

from typing import Any

PHASE15_ROOT_CAUSE_LABELS = [
    "CORPUS_GAP",
    "EXPECTED_CHUNK_RETRIEVED",
    "EXPECTED_DOCUMENT_WRONG_CHUNK",
    "RETRIEVAL_COMPETITION",
    "QUERY_MISMATCH",
    "EMBEDDING_REPRESENTATION",
    "RETRIEVAL_CONFIGURATION",
    "EVALUATION_MAPPING",
    "UNRESOLVED",
]


def classify_phase15_root_cause(
    *,
    case: dict[str, Any],
    passed_at_10: bool,
    expected_chunk_rank: int | None,
    expected_document_best_rank: int | None,
    top_100_hits: list[dict[str, Any]],
) -> str | None:
    if case.get("answerability") == "CORPUS_GAP" or case.get("expected_status") == "corpus_gap":
        return "CORPUS_GAP"

    if passed_at_10:
        return None

    if expected_chunk_rank is not None and expected_chunk_rank <= 10:
        return None

    if not case.get("expected_chunk_ids"):
        return "EVALUATION_MAPPING"

    if expected_document_best_rank is None:
        if _looks_like_query_mismatch(case.get("question", "").lower(), top_100_hits[:10]):
            return "QUERY_MISMATCH"
        return "CORPUS_GAP"

    if expected_document_best_rank <= 100 and (
        expected_chunk_rank is None or expected_chunk_rank > 10
    ):
        expected_docs = set(case["expected_document_ids"])
        expected_ids = set(case["expected_chunk_ids"])
        top10 = top_100_hits[:10]
        top10_has_expected_doc = any(hit.get("document_id") in expected_docs for hit in top10)
        top10_has_expected_chunk = any(hit.get("chunk_id") in expected_ids for hit in top10)
        if top10_has_expected_doc and not top10_has_expected_chunk:
            return "EXPECTED_DOCUMENT_WRONG_CHUNK"
        if _has_cross_document_competition(top_100_hits, case["expected_chunk_ids"], case["expected_document_ids"]):
            return "RETRIEVAL_COMPETITION"
        return "EMBEDDING_REPRESENTATION"

    if expected_document_best_rank > 100:
        if _looks_like_query_mismatch(case.get("question", "").lower(), top_100_hits[:10]):
            return "QUERY_MISMATCH"
        return "EMBEDDING_REPRESENTATION"

    return "UNRESOLVED"


def _has_cross_document_competition(
    hits: list[dict[str, Any]],
    expected_chunk_ids: list[str],
    expected_document_ids: list[str],
) -> bool:
    expected_docs = set(expected_document_ids)
    expected_ids = set(expected_chunk_ids)
    top10 = hits[:10]
    for hit in top10:
        if hit.get("chunk_id") in expected_ids:
            continue
        if hit.get("document_id") not in expected_docs:
            return True
    return False


def _looks_like_query_mismatch(query: str, top10: list[dict[str, Any]]) -> bool:
    noisy = any(token in query for token in ("who is", "deeeliver", "except", " from u"))
    if noisy:
        return True
    if not top10:
        return False
    query_terms = {term for term in query.replace("?", "").split() if len(term) > 3}
    if not query_terms:
        return False
    overlap_scores: list[float] = []
    for hit in top10:
        text = (hit.get("text") or hit.get("content") or "").lower()
        overlap = sum(1 for term in query_terms if term in text)
        overlap_scores.append(overlap / len(query_terms))
    return max(overlap_scores) < 0.2
