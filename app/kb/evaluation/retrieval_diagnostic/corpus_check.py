"""Corpus and evaluation-mapping verification helpers."""

from __future__ import annotations

import re
from typing import Any

from app.helpers.text_match import grounding_fact_matches, normalize_match_text
from app.kb.chunking.models import ProductionChunkRecord

ROOT_CAUSE_LABELS = (
    "CORPUS_GAP",
    "EVALUATION_MAPPING",
    "RETRIEVAL_CONFIGURATION",
    "CHUNKING",
    "RETRIEVAL_COMPETITION",
    "QUERY_MISMATCH",
    "EMBEDDING_REPRESENTATION",
    "UNRESOLVED",
)


def extract_knowledge_terms(expected_knowledge: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_match_text(expected_knowledge))
    return [token for token in tokens if len(token) >= 4]


def knowledge_exists_in_corpus(
    *,
    expected_knowledge: str,
    chunks: list[ProductionChunkRecord],
    extra_terms: list[str] | None = None,
) -> bool:
    combined_corpus = normalize_match_text("\n".join(chunk.content for chunk in chunks))
    if grounding_fact_matches(expected_knowledge, combined_corpus):
        return True
    terms = extract_knowledge_terms(expected_knowledge)
    if extra_terms:
        terms.extend(extra_terms)
    return any(term in combined_corpus for term in terms)


def verify_expected_chunks_exist(
    expected_chunk_ids: list[str],
    known_chunk_ids: set[str],
) -> tuple[bool, list[str]]:
    missing = [chunk_id for chunk_id in expected_chunk_ids if chunk_id not in known_chunk_ids]
    return not missing, missing


def verify_expected_mapping(
    *,
    expected_chunk_ids: list[str],
    expected_knowledge: str,
    chunk_by_id: dict[str, ProductionChunkRecord],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not expected_chunk_ids:
        return True, issues
    matched_any = False
    for chunk_id in expected_chunk_ids:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            issues.append(f"missing chunk {chunk_id}")
            continue
        combined = normalize_match_text(chunk.content)
        if grounding_fact_matches(expected_knowledge, combined):
            matched_any = True
    if not matched_any:
        issues.append("expected chunks do not contain expected knowledge text")
    return matched_any, issues


def resolve_expected_documents(case: dict[str, Any], chunk_by_id: dict[str, ProductionChunkRecord]) -> list[str]:
    document_ids: list[str] = []
    if case.get("expected_document_id"):
        document_ids.append(case["expected_document_id"])
    for chunk_id in case.get("expected_chunk_ids", []):
        chunk = chunk_by_id.get(chunk_id)
        if chunk and chunk.document_id not in document_ids:
            document_ids.append(chunk.document_id)
    return document_ids


def resolve_expected_status(
    *,
    case: dict[str, Any],
    chunks: list[ProductionChunkRecord],
    chunk_by_id: dict[str, ProductionChunkRecord],
    known_chunk_ids: set[str],
) -> dict[str, Any]:
    expected_chunk_ids = list(case.get("expected_chunk_ids", []))
    expected_document_ids = resolve_expected_documents(case, chunk_by_id)
    require_all = bool(case.get("require_all_expected_chunks", False))

    if case.get("failure_category_default") == "SOURCE_DATA" or not expected_chunk_ids:
        in_corpus = knowledge_exists_in_corpus(
            expected_knowledge=case["expected_knowledge"],
            chunks=chunks,
            extra_terms=_query_terms(case["query"]),
        )
        if not in_corpus:
            return {
                "status": "corpus_gap",
                "expected_document_ids": expected_document_ids,
                "expected_chunk_ids": expected_chunk_ids,
                "expected_needles": [case["expected_knowledge"]],
                "require_all_expected_chunks": require_all,
            }

    chunks_exist, missing = verify_expected_chunks_exist(expected_chunk_ids, known_chunk_ids)
    if expected_chunk_ids and not chunks_exist:
        return {
            "status": "mapping_issue",
            "expected_document_ids": expected_document_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_needles": [case["expected_knowledge"]],
            "require_all_expected_chunks": require_all,
            "mapping_issues": [f"missing chunks: {missing}"],
        }

    mapping_ok, mapping_issues = verify_expected_mapping(
        expected_chunk_ids=expected_chunk_ids,
        expected_knowledge=case["expected_knowledge"],
        chunk_by_id=chunk_by_id,
    )
    if expected_chunk_ids and not mapping_ok:
        return {
            "status": "mapping_issue",
            "expected_document_ids": expected_document_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_needles": [case["expected_knowledge"]],
            "require_all_expected_chunks": require_all,
            "mapping_issues": mapping_issues,
        }

    return {
        "status": "resolved",
        "expected_document_ids": expected_document_ids,
        "expected_chunk_ids": expected_chunk_ids,
        "expected_needles": [case["expected_knowledge"]],
        "require_all_expected_chunks": require_all,
    }


def _query_terms(query: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 4]
