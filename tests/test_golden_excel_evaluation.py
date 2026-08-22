"""Tests for golden Excel retrieval evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook

from app.config import settings
from app.kb.chunking.question_coverage import QUESTION_COVERAGE
from app.kb.embedding.loader import ChunkLoader
from app.kb.evaluation.golden_excel import (
    enrich_golden_cases,
    load_golden_cases_from_excel,
    resolve_expected_chunk_ids_from_answer,
    validate_golden_cases,
)
from app.helpers.text_match import grounding_fact_matches
from app.kb.evaluation.metrics import recall_at_k
from app.kb.integrity.version import KB_DATASET_VERSION


def _write_sample_excel(path: Path, rows: list[tuple[str, str, str]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Golden"
    sheet.append(["id", "question", "answer"])
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)


def test_load_golden_cases_from_excel(tmp_path: Path) -> None:
    excel_path = tmp_path / "golden.xlsx"
    _write_sample_excel(
        excel_path,
        [
            ("G001", "What is the return period for hearing aids?", "ten (10) calendar days"),
            ("G002", "Where do I ship returns?", "Sector 62, Noida"),
        ],
    )

    cases = load_golden_cases_from_excel(excel_path)
    assert len(cases) == 2
    assert cases[0]["id"] == "G001"
    assert cases[0]["query"].startswith("What is the return period")


def test_resolve_expected_chunk_ids_from_answer() -> None:
    loader = ChunkLoader(settings.chunks_dir, KB_DATASET_VERSION)
    chunks = loader.load_all().chunks
    qc01 = next(q for q in QUESTION_COVERAGE if q["id"] == "QC01")
    expected_ids, require_all = resolve_expected_chunk_ids_from_answer(
        chunks,
        answer="ten (10) calendar days",
        document_id=qc01["doc_id"],
    )
    assert expected_ids
    assert require_all is False


def test_enrich_and_validate_golden_cases(tmp_path: Path) -> None:
    loader = ChunkLoader(settings.chunks_dir, KB_DATASET_VERSION)
    chunks = loader.load_all().chunks
    qc08 = next(q for q in QUESTION_COVERAGE if q["id"] == "QC08")

    excel_path = tmp_path / "golden.xlsx"
    _write_sample_excel(
        excel_path,
        [(qc08["id"], qc08["question"], "Sector 62, Noida")],
    )
    cases, skipped = enrich_golden_cases(load_golden_cases_from_excel(excel_path), chunks)
    validate_golden_cases(cases)
    assert cases[0]["expected_chunk_ids"]
    assert isinstance(skipped, list)


def test_chunk_recall_metric() -> None:
    expected_ids = ["chunk-a"]
    retrieved = [
        {"chunk_id": "chunk-b", "content": "unrelated"},
        {"chunk_id": "chunk-a", "content": "Return shipping to Sector 62, Noida within ten days."},
    ]
    assert recall_at_k(expected_ids, [item["chunk_id"] for item in retrieved], 2) == 1.0
    assert grounding_fact_matches("Sector 62", "return shipping to sector 62, noida")
