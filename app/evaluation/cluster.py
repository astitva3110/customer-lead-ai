"""Cluster first failures. The main report is clusters, not 1000 individual rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.evaluation.classify import TurnClassification

_SEVERITY = {
    "GUARDRAIL_FAILURE": 90,
    "TOOL_FAILURE": 85,
    "LEAD_WORKFLOW_FAILURE": 80,
    "SUPPORT_WORKFLOW_FAILURE": 80,
    "STATE_FAILURE": 75,
    "GROUNDING_FAILURE": 70,
    "RERANKER_FAILURE": 65,
    "RETRIEVAL_FAILURE": 60,
    "FINAL_CONTEXT_FAILURE": 58,
    "GENERATION_FAILURE": 55,
    "QUERY_REWRITE_FAILURE": 50,
    "ROUTER_FAILURE": 48,
    "CONVERSATION_QUALITY_FAILURE": 40,
    "EXPECTED_BEHAVIOR_MISMATCH": 35,
    "CORPUS_GAP": 10,
    "NONE": 0,
}


def cluster_key(item: TurnClassification) -> str:
    pattern = item.pattern or item.failure_type
    return f"{item.failure_type}::{pattern}"


def cluster_failures(items: list[TurnClassification], *, total_conversations: int) -> list[dict[str, Any]]:
    failures = [item for item in items if not item.passed]
    groups: dict[str, list[TurnClassification]] = defaultdict(list)
    for item in failures:
        groups[cluster_key(item)].append(item)
    clusters: list[dict[str, Any]] = []
    for key, rows in groups.items():
        failure_type = rows[0].failure_type
        categories = Counter(row.category for row in rows)
        category = categories.most_common(1)[0][0]
        pattern = rows[0].pattern
        products = Counter(_value(row.product) for row in rows if row.product)
        keys = Counter(key for row in rows for key in row.knowledge_keys)
        layers = Counter(row.first_failing_layer for row in rows if row.first_failing_layer)
        retrieval_patterns = Counter(str((row.details or {}).get("raw_count", "")) for row in rows)
        selected = select_representatives(rows)
        count = len({row.conversation_id for row in rows})
        clusters.append(
            {
                "cluster_id": key,
                "failure_type": failure_type,
                "category": category,
                "count": count,
                "turn_count": len(rows),
                "percentage": round(100.0 * count / total_conversations, 2) if total_conversations else 0.0,
                "common_message_pattern": pattern,
                "common_state_pattern": pattern,
                "common_product": products.most_common(1)[0][0] if products else None,
                "common_knowledge_key": keys.most_common(1)[0][0] if keys else None,
                "common_retrieval_behavior": _retrieval_behavior(rows),
                "common_reranker_behavior": _reranker_behavior(rows),
                "first_failing_layer": layers.most_common(1)[0][0] if layers else rows[0].first_failing_layer,
                "representative_conversation_ids": [row.conversation_id for row in selected],
                "representative_trace_ids": [row.trace_id for row in selected if row.trace_id],
                "common_transition": _transition(rows),
                "severity": _SEVERITY.get(failure_type, 20),
                "retrieval_notes": dict(retrieval_patterns),
            }
        )
    clusters.sort(key=lambda item: (-item["count"], -item["severity"], item["cluster_id"]))
    return clusters


def select_representatives(rows: list[TurnClassification], *, limit: int = 5) -> list[TurnClassification]:
    if not rows:
        return []
    by_id: dict[str, TurnClassification] = {}
    for row in rows:
        by_id.setdefault(row.conversation_id, row)
    unique = list(by_id.values())
    unique.sort(key=lambda item: (-_SEVERITY.get(item.failure_type, 0), item.conversation_id))
    selected: list[TurnClassification] = []
    seen_products: set[str] = set()
    if unique:
        selected.append(unique[0])
        seen_products.add(_value(unique[0].product))
    for row in unique[1:]:
        if len(selected) >= limit:
            break
        product = _value(row.product)
        if product not in seen_products or len(selected) < 3:
            selected.append(row)
            seen_products.add(product)
    for row in unique:
        if len(selected) >= min(limit, 5):
            break
        if row.conversation_id not in {item.conversation_id for item in selected}:
            selected.append(row)
    return selected[:limit]


def _value(item: str | None) -> str:
    return str(item or "")


def _transition(rows: list[TurnClassification]) -> str | None:
    for row in rows:
        details = row.details or {}
        if details.get("from") and details.get("to"):
            return f"{details.get('from')} -> {details.get('to')}"
    return None


def _retrieval_behavior(rows: list[TurnClassification]) -> str:
    if any(row.failure_type == "RETRIEVAL_FAILURE" for row in rows):
        return "expected_evidence_missing_from_raw_candidates"
    if any(row.pattern == "EMPTY_RETRIEVAL" for row in rows):
        return "empty_candidate_set"
    return "not_primary"

def _reranker_behavior(rows: list[TurnClassification]) -> str:
    if any(row.failure_type == "RERANKER_FAILURE" for row in rows):
        return "demoted_or_dropped_available_evidence"
    return "not_primary"
