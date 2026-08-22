"""Stable expected-answer resolution for any corpus version."""

from __future__ import annotations

from app.kb.evaluation.models import EvaluationQuestion, ExpectedAnchor, ResolvedExpected
from app.kb.evaluation.retrieval_config import load_questions
from app.kb.ingestion.models import Phase12ChunkRecord
from pathlib import Path


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _chunk_haystack(chunk: Phase12ChunkRecord) -> str:
    return f"{chunk.content} {' '.join(chunk.section_path)}".lower()


def _matches_anchor(chunk: Phase12ChunkRecord, anchor: ExpectedAnchor, *, doc_labels: dict[str, str]) -> bool:
    haystack = _chunk_haystack(chunk)

    if anchor.document_id and chunk.document_id != anchor.document_id:
        return False
    if anchor.document_label:
        label_doc_id = doc_labels.get(anchor.document_label)
        if label_doc_id and chunk.document_id != label_doc_id:
            return False

    for exclude in anchor.exclude_patterns:
        if exclude.lower() in haystack:
            return False

    if anchor.required_section_prefix:
        if not any(part.strip().startswith(anchor.required_section_prefix) for part in chunk.section_path):
            return False

    for part in anchor.section_path_contains:
        if part.lower() not in haystack:
            return False

    if anchor.content_patterns and not any(p.lower() in haystack for p in anchor.content_patterns):
        return False

    if anchor.content_type and chunk.content_type != anchor.content_type:
        return False

    return True


def find_chunks_for_anchor(
    chunks: list[Phase12ChunkRecord],
    anchor: ExpectedAnchor,
    *,
    document_labels: dict[str, str] | None = None,
) -> list[Phase12ChunkRecord]:
    labels = document_labels or {}
    matched = [chunk for chunk in chunks if _matches_anchor(chunk, anchor, doc_labels=labels)]
    return sorted(matched, key=lambda item: (item.token_count, item.chunk_id))


CATALOG_SUMMARY_ONLY_KEYS: dict[str, str] = {
    "catalog:products": "Product Catalog Summary",
    "catalog:hearing_aids": "Hearing Aid Catalog Summary",
    "company:products_overview": "Product Catalog Summary",
    "hearing_aid:catalog_overview": "Hearing Aid Catalog Summary",
}

SUMMARY_SECTION_PREFERENCES: dict[str, str] = {
    "benefits:why_choose_summary": "Why Choose earKART",
    **CATALOG_SUMMARY_ONLY_KEYS,
}


def _find_catalog_summary(
    chunks: list[Phase12ChunkRecord],
    *,
    section_marker: str,
) -> list[Phase12ChunkRecord]:
    return [
        chunk
        for chunk in chunks
        if chunk.content_type == "section_summary"
        and section_marker.lower() in " ".join(chunk.section_path).lower()
    ]


def _prefer_section_summary(
    matched: list[Phase12ChunkRecord],
    *,
    section_marker: str,
) -> list[Phase12ChunkRecord]:
    summaries = [
        chunk
        for chunk in matched
        if chunk.content_type == "section_summary"
        and section_marker.lower() in " ".join(chunk.section_path).lower()
    ]
    return summaries if summaries else matched


def resolve_anchor(
    anchor: ExpectedAnchor,
    chunks: list[Phase12ChunkRecord],
    *,
    document_labels: dict[str, str] | None = None,
) -> ResolvedExpected:
    catalog_marker = CATALOG_SUMMARY_ONLY_KEYS.get(anchor.knowledge_key)
    if catalog_marker:
        matched = _find_catalog_summary(chunks, section_marker=catalog_marker)
        if not matched:
            matched = find_chunks_for_anchor(chunks, anchor, document_labels=document_labels)
    else:
        matched = find_chunks_for_anchor(chunks, anchor, document_labels=document_labels)
        section_marker = SUMMARY_SECTION_PREFERENCES.get(anchor.knowledge_key)
        if section_marker:
            matched = _prefer_section_summary(matched, section_marker=section_marker)
    if not matched:
        return ResolvedExpected(
            answerability="CORPUS_GAP",
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_section_path=[],
            knowledge_key=anchor.knowledge_key,
            mapping_notes=f"No chunk matched anchor {anchor.knowledge_key}",
        )
    best = matched[0]
    return ResolvedExpected(
        answerability="ANSWERABLE",
        expected_chunk_ids=[item.chunk_id for item in matched],
        expected_document_ids=sorted({item.document_id for item in matched}),
        expected_section_path=list(best.section_path),
        knowledge_key=anchor.knowledge_key,
        mapping_notes=f"Resolved {len(matched)} chunk(s) for {anchor.knowledge_key}",
    )


def anchor_from_dict(data: dict) -> ExpectedAnchor:
    expected = data.get("expected", data)
    return ExpectedAnchor(
        knowledge_key=expected["knowledge_key"],
        document_label=expected.get("document_label"),
        document_id=expected.get("document_id"),
        section_path_contains=list(expected.get("section_path_contains", [])),
        content_patterns=list(expected.get("content_patterns", [])),
        exclude_patterns=list(expected.get("exclude_patterns", [])),
        required_section_prefix=expected.get("required_section_prefix"),
        content_type=expected.get("content_type"),
    )


def load_evaluation_questions(path: Path | str) -> list[EvaluationQuestion]:
    rows = load_questions(path)
    questions: list[EvaluationQuestion] = []
    for row in rows:
        question_id = row.get("question_id") or row.get("id")
        if not question_id:
            raise ValueError(f"Question row missing id/question_id: {row}")
        declared = row.get("answerability")
        questions.append(
            EvaluationQuestion(
                id=question_id,
                question=row["question"],
                anchor=anchor_from_dict(row),
                category=row.get("category", ""),
                expected_answer_type=row.get("expected_answer_type", ""),
                declared_answerability=declared if declared in {"ANSWERABLE", "CORPUS_GAP"} else None,
            )
        )
    return questions


def resolve_question_expected(
    question: EvaluationQuestion,
    chunks: list[Phase12ChunkRecord],
    *,
    document_labels: dict[str, str] | None = None,
) -> ResolvedExpected:
    if question.declared_answerability == "CORPUS_GAP":
        return ResolvedExpected(
            answerability="CORPUS_GAP",
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_section_path=[],
            knowledge_key=question.anchor.knowledge_key,
            mapping_notes="Declared corpus gap in evaluation dataset",
        )
    return resolve_anchor(question.anchor, chunks, document_labels=document_labels)


def build_document_label_map(chunks: list[Phase12ChunkRecord], source_pdfs: list[str]) -> dict[str, str]:
    """Best-effort label map from filename hints."""
    labels: dict[str, str] = {}
    doc_ids = sorted({chunk.document_id for chunk in chunks})
    for doc_id in doc_ids:
        doc_chunks = [c for c in chunks if c.document_id == doc_id]
        sample = doc_chunks[0] if doc_chunks else None
        if sample is None:
            continue
        path_text = " ".join(sample.section_path).lower()
        if "terms" in path_text or doc_id.startswith("c0ca"):
            labels["terms"] = doc_id
        elif "merged" in path_text or doc_id.startswith("f9dd"):
            labels["merged"] = doc_id
    return labels
