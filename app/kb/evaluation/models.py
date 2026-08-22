"""Shared retrieval evaluation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Answerability = Literal["ANSWERABLE", "CORPUS_GAP"]


@dataclass(frozen=True)
class ExpectedAnchor:
    """Stable expected-knowledge locator — survives chunk ID changes across versions."""

    knowledge_key: str
    document_label: str | None = None
    document_id: str | None = None
    section_path_contains: list[str] = field(default_factory=list)
    content_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    required_section_prefix: str | None = None
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_key": self.knowledge_key,
            "document_label": self.document_label,
            "document_id": self.document_id,
            "section_path_contains": self.section_path_contains,
            "content_patterns": self.content_patterns,
            "exclude_patterns": self.exclude_patterns,
            "required_section_prefix": self.required_section_prefix,
            "content_type": self.content_type,
        }


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    anchor: ExpectedAnchor
    category: str = ""
    expected_answer_type: str = ""
    declared_answerability: Answerability | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "question": self.question,
            "expected": self.anchor.to_dict(),
        }
        if self.category:
            payload["category"] = self.category
        if self.expected_answer_type:
            payload["expected_answer_type"] = self.expected_answer_type
        if self.declared_answerability:
            payload["answerability"] = self.declared_answerability
        return payload


@dataclass
class ResolvedExpected:
    answerability: Answerability
    expected_chunk_ids: list[str]
    expected_document_ids: list[str]
    expected_section_path: list[str]
    knowledge_key: str
    mapping_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerability": self.answerability,
            "expected_chunk_ids": self.expected_chunk_ids,
            "expected_document_ids": self.expected_document_ids,
            "expected_section_path": self.expected_section_path,
            "knowledge_key": self.knowledge_key,
            "mapping_notes": self.mapping_notes,
        }


@dataclass
class RankedHit:
    rank: int
    similarity: float
    chunk_id: str
    document_id: str
    section_path: list[str]
    token_count: int
    content_type: str
    text: str
    page_number: int | None = None
    document_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "similarity": self.similarity,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "page_number": self.page_number,
            "section_path": self.section_path,
            "content_type": self.content_type,
            "token_count": self.token_count,
            "text": self.text,
        }


@dataclass
class QueryEvaluationResult:
    id: str
    question: str
    answerability: Answerability
    expected: ResolvedExpected
    expected_chunk_rank: int | None
    expected_document_best_rank: int | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    passed_at_1: bool
    passed_at_3: bool
    passed_at_5: bool
    passed_at_10: bool
    root_cause: str | None
    top_results: list[RankedHit]
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    category: str = ""
    expected_answer_type: str = ""
    failure_analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "expected_answer_type": self.expected_answer_type,
            "answerability": self.answerability,
            "expected_chunk_ids": self.expected.expected_chunk_ids,
            "expected_document_ids": self.expected.expected_document_ids,
            "expected_section_path": self.expected.expected_section_path,
            "expected_chunk_rank": self.expected_chunk_rank,
            "expected_document_best_rank": self.expected_document_best_rank,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "passed_at_1": self.passed_at_1,
            "passed_at_3": self.passed_at_3,
            "passed_at_5": self.passed_at_5,
            "passed_at_10": self.passed_at_10,
            "root_cause": self.root_cause,
            "mapping_notes": self.expected.mapping_notes,
            "knowledge_key": self.expected.knowledge_key,
            "top_results": [hit.to_dict() for hit in self.top_results],
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
        }
        if self.failure_analysis:
            payload["failure_analysis"] = self.failure_analysis
        return payload
