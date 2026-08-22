"""Frozen retrieval evaluation dataset qc35_v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.chunking.question_coverage import QUESTION_COVERAGE
from app.kb.chunking.semantic_units import effective_document_type

EVALUATION_DATASET_VERSION = "qc35_v1"
DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "qc35_v1.json"


def _scope_chunks(chunks: list[ProductionChunkRecord], q: dict) -> list[ProductionChunkRecord]:
    scope = chunks
    if q.get("doc_id"):
        scope = [c for c in chunks if c.document_id == q["doc_id"]]
    if q.get("doc_type"):
        scope = [
            c
            for c in chunks
            if effective_document_type(c.document_type, c.title, c.canonical_url) == q["doc_type"]
        ]
    if q.get("extraction_method"):
        scope = [c for c in scope if c.extraction_method.value == q["extraction_method"]]
    return scope


def resolve_expected_chunk_ids(chunks: list[ProductionChunkRecord], q: dict) -> tuple[list[str], bool]:
    """Return expected chunk IDs and whether all of them must appear in top-k."""
    scope = _scope_chunks(chunks, q)
    needles = q.get("needles", [])

    matched: list[str] = []
    for chunk in scope:
        if all(needle.lower() in chunk.content.lower() for needle in needles):
            matched.append(chunk.chunk_id)
    if matched:
        if q.get("require_single_chunk"):
            return [matched[0]], True
        return matched, False

    if len(needles) > 1:
        per_needle: list[str] = []
        for needle in needles:
            for chunk in scope:
                if needle.lower() in chunk.content.lower():
                    per_needle.append(chunk.chunk_id)
                    break
        if len(per_needle) == len(needles):
            unique_ids = list(dict.fromkeys(per_needle))
            return unique_ids, len(unique_ids) > 1

    return [], False


def build_evaluation_dataset(chunks: list[ProductionChunkRecord]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for q in QUESTION_COVERAGE:
        expected_chunk_ids, require_all_expected_chunks = resolve_expected_chunk_ids(chunks, q)
        cases.append(
            {
                "id": q["id"],
                "evaluation_id": q["id"],
                "query": q["question"],
                "expected_document_id": q.get("doc_id"),
                "expected_chunk_ids": expected_chunk_ids,
                "expected_needles": q.get("needles", []),
                "category": q.get("topic", "general"),
                "importance": "critical" if q["id"] in {"QC08", "QC13", "QC14", "QC15", "QC16"} else "standard",
                "require_single_chunk": q.get("require_single_chunk", False),
                "require_all_expected_chunks": require_all_expected_chunks,
            }
        )
    return cases


def load_or_build_dataset(chunks: list[ProductionChunkRecord], *, force_rebuild: bool = False) -> list[dict[str, Any]]:
    dataset = build_evaluation_dataset(chunks)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def validate_dataset(dataset: list[dict[str, Any]]) -> None:
    missing = [case["id"] for case in dataset if not case.get("expected_chunk_ids")]
    if missing:
        raise ValueError(
            f"Evaluation dataset has cases without resolvable expected_chunk_ids: {missing[:10]}"
        )
