"""Root-cause classification for retrieval diagnostics."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.retrieval_diagnostic.corpus_check import ROOT_CAUSE_LABELS
from app.kb.evaluation.retrieval_diagnostic.ranking import top_competing_chunks


def _rank10_similarity(vector_hits: list[dict[str, Any]]) -> float | None:
    for hit in vector_hits:
        if hit["rank"] == 10:
            return hit.get("similarity")
    return vector_hits[-1].get("similarity") if vector_hits else None


def classify_root_cause(
    *,
    case: dict[str, Any],
    expected: dict[str, Any],
    chunk_analysis: dict[str, Any],
    document_analysis: dict[str, Any],
    keyword_analysis: dict[str, Any],
    whole_document_analysis: dict[str, Any],
    vector_hits: list[dict[str, Any]],
    retrieval_config_issues: list[str] | None = None,
) -> dict[str, Any]:
    signals = build_signals(
        chunk_analysis=chunk_analysis,
        document_analysis=document_analysis,
        keyword_analysis=keyword_analysis,
        whole_document_analysis=whole_document_analysis,
        vector_hits=vector_hits,
        expected_chunk_ids=set(expected.get("expected_chunk_ids", [])),
    )
    evidence: list[str] = []
    secondary: list[str] = []

    if retrieval_config_issues:
        evidence.extend(retrieval_config_issues)
        return _diagnosis(
            primary="RETRIEVAL_CONFIGURATION",
            secondary=[],
            confidence=0.9,
            signals=signals,
            evidence=evidence,
            recommendation="Verify vector table, embedding version filters, and similarity metric configuration.",
        )

    if expected.get("status") == "corpus_gap":
        evidence.append("Required knowledge was not found anywhere in the frozen corpus.")
        return _diagnosis(
            primary="CORPUS_GAP",
            secondary=secondary,
            confidence=0.95,
            signals=signals,
            evidence=evidence,
            recommendation="Add source content to the KB or exclude this question from retrieval evaluation.",
        )

    if expected.get("status") == "mapping_issue":
        issues = expected.get("mapping_issues") or ["Expected mapping could not be verified."]
        evidence.extend(issues)
        return _diagnosis(
            primary="EVALUATION_MAPPING",
            secondary=secondary,
            confidence=0.9,
            signals=signals,
            evidence=evidence,
            recommendation="Fix evaluation ground truth before interpreting retrieval metrics.",
        )

    best_chunk_rank = chunk_analysis.get("best_expected_chunk_rank")
    best_doc_rank = document_analysis.get("expected_document_best_rank")
    keyword_chunk_rank = keyword_analysis.get("expected_chunk_best_rank")
    whole_doc_sim = whole_document_analysis.get("similarity")
    best_chunk_sim = chunk_analysis.get("best_expected_chunk_similarity")

    if best_chunk_rank is not None and best_chunk_rank <= 10:
        evidence.append(f"Expected knowledge chunk reached rank {best_chunk_rank} in vector search.")
        return _diagnosis(
            primary="UNRESOLVED",
            secondary=["RETRIEVAL_COMPETITION"] if best_chunk_rank > 1 else [],
            confidence=0.85,
            signals=signals,
            evidence=evidence,
            recommendation="Retrieval succeeded at top-10; no corrective action required for this question.",
        )

    if (
        best_doc_rank is not None
        and best_doc_rank <= 10
        and (best_chunk_rank is None or best_chunk_rank > 10)
    ):
        evidence.append(
            f"Expected document chunk rank={best_doc_rank} but knowledge chunk rank={best_chunk_rank or '>100'}."
        )
        signals["chunking_signal"] = True
        return _diagnosis(
            primary="CHUNKING",
            secondary=["RETRIEVAL_COMPETITION"] if signals["semantic_competition_detected"] else [],
            confidence=0.9,
            signals=signals,
            evidence=evidence,
            recommendation="Improve chunk boundaries on the expected document so answer-bearing text is retrievable.",
        )

    if signals["semantic_competition_detected"]:
        best_sim = chunk_analysis.get("best_expected_chunk_similarity")
        evidence.append(
            f"Expected chunk is semantically close (rank={best_chunk_rank}, similarity={best_sim}) "
            "but outranked by competing chunks."
        )
        competing = top_competing_chunks(
            vector_hits,
            expected_chunk_ids=set(expected.get("expected_chunk_ids", [])),
            expected_document_ids=set(expected.get("expected_document_ids", [])),
        )
        if competing:
            evidence.append(
                "Top competing chunks: "
                + ", ".join(f"#{item['rank']} sim={item.get('similarity')}" for item in competing[:3])
            )
        return _diagnosis(
            primary="RETRIEVAL_COMPETITION",
            secondary=["CHUNKING"] if signals["chunking_signal"] else [],
            confidence=0.85,
            signals=signals,
            evidence=evidence,
            recommendation="Consider reranking or document-aware retrieval; do not assume embedding model failure.",
        )

    if signals["keyword_beats_vector"] and (
        best_chunk_rank is None or best_chunk_rank > 50
    ) and (
        best_doc_rank is None or best_doc_rank > 50
    ):
        if case.get("query_note") and keyword_chunk_rank is not None and keyword_chunk_rank <= 20:
            evidence.append(case["query_note"])
            evidence.append("Lexical retrieval finds related content while vector retrieval does not.")
            return _diagnosis(
                primary="QUERY_MISMATCH",
                secondary=["EMBEDDING_REPRESENTATION"],
                confidence=0.75,
                signals=signals,
                evidence=evidence,
                recommendation="Test query reformulations and alias handling before changing the embedding model.",
            )
        evidence.append(
            f"Lexical BM25 found expected chunk at rank {keyword_chunk_rank}, "
            f"while vector chunk rank={best_chunk_rank or '>100'}."
        )
        if whole_doc_sim is not None and best_chunk_sim is not None and whole_doc_sim <= best_chunk_sim + 0.02:
            evidence.append("Whole-document semantic similarity did not exceed best chunk similarity.")
        signals["embedding_signal"] = True
        return _diagnosis(
            primary="EMBEDDING_REPRESENTATION",
            secondary=[],
            confidence=0.8,
            signals=signals,
            evidence=evidence,
            recommendation="Investigate query/document embedding alignment before changing production retrieval config.",
        )


    if signals["chunking_signal"]:
        evidence.append("Expected document appears in top ranks but knowledge-bearing chunk does not.")
        return _diagnosis(
            primary="CHUNKING",
            secondary=["RETRIEVAL_COMPETITION"] if signals["semantic_competition_detected"] else [],
            confidence=0.8,
            signals=signals,
            evidence=evidence,
            recommendation="Refine chunking on the expected document to keep headings with answer text.",
        )

    if best_chunk_rank is None and keyword_chunk_rank is None:
        evidence.append("Expected knowledge chunk was not found in vector top-100 or BM25 top-100.")
        if expected.get("status") == "resolved":
            evidence.append("Knowledge exists in corpus but retrieval did not surface mapped chunks.")
        return _diagnosis(
            primary="UNRESOLVED",
            secondary=["EMBEDDING_REPRESENTATION"] if signals["embedding_signal"] else [],
            confidence=0.55,
            signals=signals,
            evidence=evidence,
            recommendation="Collect additional evidence (manual chunk review, alternate mappings) before changing models.",
        )

    evidence.append("Evidence is mixed; no single dominant failure mode was detected.")
    return _diagnosis(
        primary="UNRESOLVED",
        secondary=[],
        confidence=0.5,
        signals=signals,
        evidence=evidence,
        recommendation="Review top-100 hits manually and refine diagnostic thresholds if needed.",
    )


def build_signals(
    *,
    chunk_analysis: dict[str, Any],
    document_analysis: dict[str, Any],
    keyword_analysis: dict[str, Any],
    whole_document_analysis: dict[str, Any],
    vector_hits: list[dict[str, Any]],
    expected_chunk_ids: set[str],
) -> dict[str, bool]:
    best_chunk_rank = chunk_analysis.get("best_expected_chunk_rank")
    best_chunk_sim = chunk_analysis.get("best_expected_chunk_similarity")
    keyword_chunk_rank = keyword_analysis.get("expected_chunk_best_rank")
    whole_doc_sim = whole_document_analysis.get("similarity")

    keyword_beats_vector = False
    if keyword_chunk_rank is not None:
        if best_chunk_rank is None:
            keyword_beats_vector = keyword_chunk_rank <= 20
        else:
            keyword_beats_vector = keyword_chunk_rank + 10 <= best_chunk_rank

    whole_document_beats_best_chunk = False
    if whole_doc_sim is not None and best_chunk_sim is not None:
        whole_document_beats_best_chunk = whole_doc_sim > best_chunk_sim + 0.01

    rank10_sim = _rank10_similarity(vector_hits)
    semantic_competition_detected = False
    if best_chunk_rank is not None and best_chunk_rank > 10 and best_chunk_sim is not None:
        if rank10_sim is not None and best_chunk_sim >= rank10_sim - 0.015:
            semantic_competition_detected = True
        elif best_chunk_rank <= 30 and best_chunk_sim >= 0.70:
            semantic_competition_detected = True

    doc_rank = document_analysis.get("expected_document_best_rank")
    chunking_signal = bool(
        doc_rank is not None
        and doc_rank <= 20
        and (best_chunk_rank is None or best_chunk_rank > 20)
    )

    embedding_signal = bool(
        keyword_beats_vector
        and (best_chunk_rank is None or best_chunk_rank > 50)
    )

    return {
        "expected_chunk_in_top_10": chunk_analysis.get("expected_chunks_in_top_10", False),
        "expected_chunk_in_top_20": chunk_analysis.get("expected_chunks_in_top_20", False),
        "expected_chunk_in_top_50": chunk_analysis.get("expected_chunks_in_top_50", False),
        "expected_chunk_in_top_100": chunk_analysis.get("expected_chunks_in_top_100", False),
        "expected_document_in_top_10": document_analysis.get("expected_document_in_top_10", False),
        "expected_document_in_top_20": document_analysis.get("expected_document_in_top_20", False),
        "expected_document_in_top_50": document_analysis.get("expected_document_in_top_50", False),
        "expected_document_in_top_100": document_analysis.get("expected_document_in_top_100", False),
        "keyword_beats_vector": keyword_beats_vector,
        "whole_document_beats_best_chunk": whole_document_beats_best_chunk,
        "semantic_competition_detected": semantic_competition_detected,
        "chunking_signal": chunking_signal,
        "embedding_signal": embedding_signal,
        "corpus_gap_signal": False,
        "evaluation_mapping_signal": False,
    }


def _diagnosis(
    *,
    primary: str,
    secondary: list[str],
    confidence: float,
    signals: dict[str, bool],
    evidence: list[str],
    recommendation: str,
) -> dict[str, Any]:
    if primary == "CORPUS_GAP":
        signals["corpus_gap_signal"] = True
    if primary == "EVALUATION_MAPPING":
        signals["evaluation_mapping_signal"] = True
    valid_secondary = [label for label in secondary if label in ROOT_CAUSE_LABELS and label != primary]
    return {
        "primary_root_cause": primary,
        "secondary_root_causes": valid_secondary,
        "confidence": round(confidence, 2),
        "signals": signals,
        "evidence": evidence,
        "recommendation": recommendation,
    }
