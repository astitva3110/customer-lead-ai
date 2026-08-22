"""Lightweight root-cause classification (vector-only, no BM25/whole-doc)."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.retrieval_diagnostic.ranking import top_competing_chunks


def _rank10_similarity(hits: list[dict[str, Any]]) -> float | None:
    for hit in hits:
        if hit["rank"] == 10:
            return hit.get("similarity")
    return hits[-1].get("similarity") if hits else None


def classify_lightweight_root_cause(
    *,
    case: dict[str, Any],
    expected_status: str,
    mapping_issues: list[str],
    chunk_rank: int | None,
    chunk_similarity: float | None,
    document_rank: int | None,
    document_similarity: float | None,
    top_hits: list[dict[str, Any]],
    expected_chunk_ids: set[str],
    expected_document_ids: set[str],
    retrieval_config_issues: list[str] | None = None,
) -> dict[str, Any]:
    evidence: list[str] = []

    if retrieval_config_issues:
        evidence.extend(retrieval_config_issues)
        return _result(
            "RETRIEVAL_CONFIGURATION",
            0.9,
            evidence,
            "Verify vector table, embedding version filter, and similarity metric configuration.",
        )

    if expected_status == "corpus_gap":
        evidence.append("Required knowledge was not found in the frozen KB corpus.")
        return _result(
            "CORPUS_GAP",
            0.95,
            evidence,
            "Add source content or exclude this question from retrieval evaluation.",
        )

    if expected_status == "mapping_issue":
        evidence.extend(mapping_issues or ["Expected mapping could not be verified."])
        return _result(
            "EVALUATION_MAPPING",
            0.9,
            evidence,
            "Fix evaluation ground truth before interpreting retrieval metrics.",
        )

    if chunk_rank is not None and chunk_rank <= 10:
        evidence.append(f"Expected knowledge chunk reached rank {chunk_rank}.")
        return _result(
            "UNRESOLVED",
            0.9,
            evidence,
            "Retrieval succeeded at top-10; no failure mode detected.",
        )

    if (
        document_rank is not None
        and document_rank <= 10
        and (chunk_rank is None or chunk_rank > 10)
    ):
        evidence.append(
            f"Expected document best chunk rank={document_rank} but knowledge chunk rank={chunk_rank or '>100'}."
        )
        return _result(
            "CHUNKING",
            0.9,
            evidence,
            "Improve chunk boundaries so answer-bearing text is retrievable from the expected document.",
        )

    rank10_sim = _rank10_similarity(top_hits)
    if (
        chunk_rank is not None
        and 11 <= chunk_rank <= 30
        and chunk_similarity is not None
        and rank10_sim is not None
        and chunk_similarity >= rank10_sim - 0.015
    ):
        evidence.append(
            f"Expected chunk rank={chunk_rank} (sim={chunk_similarity}) is close to top-10 cutoff (sim={rank10_sim})."
        )
        competing = top_competing_chunks(
            top_hits,
            expected_chunk_ids=expected_chunk_ids,
            expected_document_ids=expected_document_ids,
            limit=3,
        )
        if competing:
            evidence.append(
                "Top competing: "
                + ", ".join(f"#{c['rank']} sim={c.get('similarity')}" for c in competing)
            )
        return _result(
            "RETRIEVAL_COMPETITION",
            0.85,
            evidence,
            "Consider reranking or document-aware retrieval; do not assume embedding model failure.",
        )

    if case.get("query_note") and chunk_rank is None:
        evidence.append(case["query_note"])
        evidence.append("Expected chunk not found in top-100; query terminology may differ from source text.")
        return _result(
            "QUERY_MISMATCH",
            0.7,
            evidence,
            "Test query reformulations and alias handling before changing the embedding model.",
        )

    if (
        document_rank is not None
        and document_rank <= 20
        and (chunk_rank is None or chunk_rank > 20)
    ):
        evidence.append(
            f"Expected document appears at rank {document_rank} but knowledge chunk rank={chunk_rank or '>100'}."
        )
        return _result(
            "CHUNKING",
            0.8,
            evidence,
            "Useful content may be fragmented or buried in low-information chunks on the same page.",
        )

    if chunk_rank is None and document_rank is None:
        evidence.append("Expected chunk and document were not found in vector top-100.")
        return _result(
            "UNRESOLVED",
            0.55,
            evidence,
            "Review top-100 metadata manually; insufficient evidence for embedding failure.",
        )

    evidence.append("Evidence is mixed; no dominant failure mode identified from vector ranks alone.")
    return _result(
        "UNRESOLVED",
        0.5,
        evidence,
        "Inspect top-10 text and chunk boundaries before changing models or retrieval config.",
    )


def _result(primary: str, confidence: float, evidence: list[str], recommendation: str) -> dict[str, Any]:
    return {
        "primary_root_cause": primary,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "recommendation": recommendation,
    }
