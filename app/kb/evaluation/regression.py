"""Critical regression retrieval checks."""

from __future__ import annotations

from typing import Any

from app.kb.chunking.fidelity import (
    POLICY_CLAUSE_CHECKS,
    PRODUCT_SPEC_CHECKS,
    RETURNS_DOC_ID,
    RADIUS_M16_DOC_ID,
)
from app.kb.vector.search import VectorSearchService

REGRESSION_QUERIES = [
    {
        "id": "REG_POLICY",
        "query": "Return and replacement policy clauses eligibility refund statutory rights",
        "document_id": RETURNS_DOC_ID,
        "needles": list(POLICY_CLAUSE_CHECKS),
    },
    {
        "id": "REG_PRODUCT_BATTERY_LIFE",
        "query": "Radius M16 BTE battery life",
        "document_id": RADIUS_M16_DOC_ID,
        "needles": ["270 Hrs"],
    },
    {
        "id": "REG_PRODUCT_BATTERY_SIZE",
        "query": "Radius M16 BTE battery size",
        "document_id": RADIUS_M16_DOC_ID,
        "needles": ["Battery Size", "13"],
    },
    {
        "id": "REG_PRODUCT_ATTACK",
        "query": "Radius M16 attack time",
        "document_id": RADIUS_M16_DOC_ID,
        "needles": ["Attack Time 28 ms"],
    },
    {
        "id": "REG_PRODUCT_RELEASE",
        "query": "Radius M16 release time",
        "document_id": RADIUS_M16_DOC_ID,
        "needles": ["Release Time 891 ms"],
    },
    {
        "id": "REG_FAQ",
        "query": "FAQ battery life question and answer",
        "needles": ["Question:", "Answer:", "battery"],
    },
    {
        "id": "REG_OCR",
        "query": "Due Diligence Certificate SEBI prospectus certification",
        "needles": ["Due Diligence Certificate", "SEBI"],
    },
    {
        "id": "REG_INVESTOR",
        "query": "IPO prospectus risk factors equity investments",
        "needles": ["risk factors", "Investments in equity"],
    },
    {
        "id": "REG_PROSPECTUS",
        "query": "Prospectus section general manufacturing facility Noida",
        "needles": ["SECTION", "Manufacturing Facility"],
    },
]


def run_regression_checks(search: VectorSearchService, *, top_k: int = 10) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in REGRESSION_QUERIES:
        filters = {}
        if case.get("document_id"):
            filters["document_id"] = case["document_id"]
        hits = search.search(case["query"], top_k=top_k, **filters)
        combined = "\n".join(hit["content"] for hit in hits)
        needle_results = {
            needle: needle.lower() in combined.lower() for needle in case["needles"]
        }
        passed = all(needle_results.values()) and bool(hits)
        metadata_leak = any("sha256:" in hit.get("content", "").lower() for hit in hits[:3])
        results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "passed": passed and not metadata_leak,
                "needle_results": needle_results,
                "metadata_leak": metadata_leak,
                "top_chunk_ids": [hit["chunk_id"] for hit in hits[:3]],
            }
        )
    return {
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
