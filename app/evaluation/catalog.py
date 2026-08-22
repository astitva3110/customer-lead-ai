"""Knowledge catalog from the existing resolver dataset. No second knowledge-key system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.kb.evaluation.models import ExpectedAnchor
from app.kb.evaluation.resolver.expected import anchor_from_dict, load_evaluation_questions
from app.kb.evaluation.retrieval_config import load_questions


@dataclass(frozen=True)
class CatalogItem:
    question_id: str
    question: str
    category: str
    expected_answer_type: str
    answerable: bool
    knowledge_key: str
    anchor: ExpectedAnchor
    retrieval_bucket: str

    def to_anchor_dict(self) -> dict[str, Any]:
        return self.anchor.to_dict()


_BUCKETS = {
    "PRODUCT": "Product",
    "SPECIFICATION": "Specification",
    "POLICY": "Policy",
    "HEARING_AID": "Hearing Aid",
    "COMPANY": "Company",
    "SUMMARY": "Summary",
    "NOISY_QUERY": "Noisy Query",
    "CORPUS_GAP": "Specification",
}


def retrieval_bucket_for(category: str, conversation_category: str | None = None) -> str:
    if conversation_category == "KNOWLEDGE_FOLLOWUP":
        return "Follow-up"
    if conversation_category == "NOISY_QUERY":
        return "Noisy Query"
    return _BUCKETS.get((category or "").upper(), "Product")


def load_knowledge_catalog(path: Path | str) -> list[CatalogItem]:
    rows = load_questions(path)
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id") or row.get("id") or "")
        if not question_id or question_id in seen:
            continue
        seen.add(question_id)
        expected = row.get("expected") or {}
        knowledge_key = str(expected.get("knowledge_key") or "")
        if not knowledge_key:
            continue
        answerable = str(row.get("answerability") or "ANSWERABLE") != "CORPUS_GAP"
        category = str(row.get("category") or "PRODUCT")
        items.append(
            CatalogItem(
                question_id=question_id,
                question=str(row["question"]),
                category=category,
                expected_answer_type=str(row.get("expected_answer_type") or ""),
                answerable=answerable,
                knowledge_key=knowledge_key,
                anchor=anchor_from_dict(row),
                retrieval_bucket=retrieval_bucket_for(category),
            )
        )
    if not items:
        raise ValueError(f"knowledge catalog is empty: {path}")
    return items


def answerable_items(catalog: list[CatalogItem]) -> list[CatalogItem]:
    return [item for item in catalog if item.answerable]


def corpus_gap_items(catalog: list[CatalogItem]) -> list[CatalogItem]:
    return [item for item in catalog if not item.answerable]


def items_by_key(catalog: list[CatalogItem]) -> dict[str, CatalogItem]:
    mapping: dict[str, CatalogItem] = {}
    for item in catalog:
        mapping.setdefault(item.knowledge_key, item)
    return mapping


def items_for_product(catalog: list[CatalogItem], product: str) -> list[CatalogItem]:
    needle = product.lower()
    return [
        item
        for item in catalog
        if item.answerable and (needle in item.question.lower() or needle in item.knowledge_key.lower())
    ]


def load_catalog_questions_for_resolver(path: Path | str):
    """Existing evaluation question objects — reuse resolver, do not recreate it."""
    return load_evaluation_questions(path)
