"""Verified ground truth only: catalog knowledge_keys + explicit deterministic rules.

Scenario fields produced by the conversation generator are hypotheses, not GT.
"""

from __future__ import annotations

from typing import Any

from app.services.conversation.guardrail import guardrail_check
from app.services.conversation.query_rewriter import extract_product, needs_rewrite, should_bind_product
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.evaluation.catalog import CatalogItem
from app.evaluation.evidence import evidence_in_rows, expected_items_for_keys, matching_ranks

SOURCE_CATALOG = "verified_knowledge_key"
SOURCE_RULE = "deterministic_rule"


def verified_items(keys: list[str], catalog: dict[str, CatalogItem]) -> list[CatalogItem]:
    return expected_items_for_keys(list(keys or []), catalog)


def verified_answerable(items: list[CatalogItem]) -> list[CatalogItem]:
    return [item for item in items if item.answerable]


def verified_corpus_gap(items: list[CatalogItem]) -> list[CatalogItem]:
    return [item for item in items if not item.answerable]


def expected_blocked(message: str) -> str | None:
    """Production guardrail rule applied to the actual user text."""
    return guardrail_check(message or "")


def named_product(message: str) -> str:
    return extract_product(message or "")


def rewrite_required(message: str, product_before: str | None) -> bool:
    if not product_before:
        return False
    text = message or ""
    return needs_rewrite(text, product_before) or should_bind_product(text, product_before)


def is_insufficient(response_text: str, grounding: dict[str, Any]) -> bool:
    if INSUFFICIENT_INFORMATION_MESSAGE.lower() in (response_text or "").lower():
        return True
    if grounding.get("fallback_triggered") and grounding.get("grounded_returned") is not True:
        return True
    if grounding.get("validator_reason") in {"llm_ungrounded", "empty_hits"}:
        return True
    return False


def grounded_ok(grounding: dict[str, Any]) -> bool:
    return (
        bool(grounding.get("grounded_returned"))
        and bool(grounding.get("validator_result"))
        and not grounding.get("fallback_triggered")
    )


def evidence_ranks(rows: list[dict[str, Any]], items: list[CatalogItem]) -> list[int]:
    return matching_ranks(rows, items)


def has_evidence(rows: list[dict[str, Any]], items: list[CatalogItem]) -> bool:
    return evidence_in_rows(rows, items) if items else False


def norm_product(value: Any) -> str:
    return " ".join(str(value or "").lower().split())
