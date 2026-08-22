"""Conversation/retrieval/latency/LLM metrics from ChatTrace. Measurement only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.evaluation.catalog import CatalogItem
from app.evaluation.classify import TurnClassification
from app.evaluation.evidence import (
    matching_ranks,
    retrieval_metrics_from_ids,
    synthetic_expected_ids,
)
from app.evaluation.ground_truth import verified_answerable, verified_items
from app.evaluation.schema import GeneratedConversation, TurnMessage
from app.kb.evaluation.metrics import aggregate_metrics


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def latency_percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def turn_latency_ms(trace: dict[str, Any]) -> float | None:
    latency = trace.get("latency") or {}
    value = latency.get("total_ms")
    if value is None:
        return None
    return float(value)


def llm_calls_from_trace(trace: dict[str, Any]) -> dict[str, int]:
    generation = trace.get("generation") or {}
    query = trace.get("query") or {}
    turn = trace.get("turn_understanding") or {}
    gen_calls = int(generation.get("llm_call_count") or (1 if generation.get("model") else 0))
    rewrite_calls = 1 if query.get("rewrite_executed") else 0
    router_calls = 1 if turn.get("method") == "llm" else 0
    return {
        "generation": gen_calls,
        "query_rewrite": rewrite_calls,
        "router": router_calls,
        "total": gen_calls + router_calls,
    }


def rerank_delta(trace: dict[str, Any], items: list[CatalogItem]) -> dict[str, Any] | None:
    raw = list((trace.get("retrieval") or {}).get("candidates") or [])
    reranked = list((trace.get("reranking") or {}).get("candidates") or [])
    if not raw or not items:
        return None
    raw_ranks = matching_ranks(raw, items)
    rerank_ranks = matching_ranks(reranked, items)
    raw_rank = raw_ranks[0] if raw_ranks else None
    rerank_rank = rerank_ranks[0] if rerank_ranks else None
    delta = None
    if raw_rank is not None and rerank_rank is not None:
        delta = raw_rank - rerank_rank
    status = "unchanged"
    if raw_rank is not None and rerank_rank is not None:
        if rerank_rank < raw_rank:
            status = "improved"
        elif rerank_rank > raw_rank:
            status = "regressed"
    elif raw_rank is not None and rerank_rank is None:
        status = "regressed"
    return {
        "raw_rank": raw_rank,
        "reranked_rank": rerank_rank,
        "rank_delta": delta,
        "status": status,
    }


def retrieval_turn_metrics(
    message: TurnMessage,
    trace: dict[str, Any],
    catalog: dict[str, CatalogItem],
) -> dict[str, Any] | None:
    items = verified_answerable(verified_items(message.knowledge_keys, catalog))
    raw = list((trace.get("retrieval") or {}).get("candidates") or [])
    if not items or not raw:
        return None
    retrieved_ids = [str(row.get("chunk_id")) for row in raw if row.get("chunk_id")]
    ranks = matching_ranks(raw, items)
    expected_ids = synthetic_expected_ids(ranks, retrieved_ids)
    metrics = retrieval_metrics_from_ids(expected_ids, retrieved_ids)
    metrics["passed"] = metrics["recall_at_10"] == 1.0
    metrics["bucket"] = message.retrieval_bucket or "Product"
    metrics["category"] = message.retrieval_bucket or "Product"
    metrics["ground_truth_source"] = "verified_knowledge_key"
    return metrics


def aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapped = [
        {
            "recall_at_1": row["recall_at_1"],
            "recall_at_3": row["recall_at_3"],
            "recall_at_5": row["recall_at_5"],
            "recall_at_10": row["recall_at_10"],
            "mrr": row["mrr"],
            "passed": row.get("passed", row["recall_at_10"] == 1.0),
        }
        for row in rows
    ]
    summary = aggregate_metrics(mapped)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row.get("bucket") or "Product")].append(row)
    summary["by_category"] = {key: aggregate_metrics([
        {
            "recall_at_1": item["recall_at_1"],
            "recall_at_3": item["recall_at_3"],
            "recall_at_5": item["recall_at_5"],
            "recall_at_10": item["recall_at_10"],
            "mrr": item["mrr"],
            "passed": item.get("passed", item["recall_at_10"] == 1.0),
        }
        for item in values
    ]) for key, values in sorted(by_bucket.items())}
    return summary


def layer_metrics(turn_results: list[TurnClassification]) -> dict[str, Any]:
    layers = {
        "guardrail": "GUARDRAIL_FAILURE",
        "router": "ROUTER_FAILURE",
        "query_rewrite": "QUERY_REWRITE_FAILURE",
        "retrieval": "RETRIEVAL_FAILURE",
        "reranker": "RERANKER_FAILURE",
        "generation": "GENERATION_FAILURE",
        "grounding": "GROUNDING_FAILURE",
        "state": "STATE_FAILURE",
        "lead": "LEAD_WORKFLOW_FAILURE",
        "support": "SUPPORT_WORKFLOW_FAILURE",
        "tools": "TOOL_FAILURE",
    }
    payload: dict[str, Any] = {}
    for layer, failure in layers.items():
        failed = sum(1 for item in turn_results if item.failure_type == failure)
        payload[layer] = {"candidate_failures": failed, "failures": failed}
    final_failures = sum(1 for item in turn_results if item.failure_type == "FINAL_CONTEXT_FAILURE")
    payload["final_context"] = {"candidate_failures": final_failures, "failures": final_failures}
    return payload


def category_metrics(
    conversations: list[GeneratedConversation],
    conversation_results: list[TurnClassification],
) -> dict[str, Any]:
    by_id = {item.conversation_id: item for item in conversation_results}
    grouped: dict[str, list[GeneratedConversation]] = defaultdict(list)
    for conversation in conversations:
        grouped[conversation.category].append(conversation)
    payload: dict[str, Any] = {}
    for category, rows in sorted(grouped.items()):
        candidates = sum(1 for item in rows if by_id.get(item.conversation_id) and by_id[item.conversation_id].candidate)
        payload[category] = {
            "count": len(rows),
            "candidate_failures": candidates,
        }
    return payload


def conversation_behavior_metrics(
    conversations: list[GeneratedConversation],
    per_conversation: list[dict[str, Any]],
) -> dict[str, Any]:
    rag_turns = 0
    lead_conversations = 0
    support_conversations = 0
    product_switch_conversations = 0
    for conversation, result in zip(conversations, per_conversation):
        turns = result.get("turns") or []
        rag_turns += sum(1 for item in turns if item.get("retrieval_used"))
        if conversation.category.startswith("LEAD"):
            lead_conversations += 1
        if conversation.category.startswith("SUPPORT"):
            support_conversations += 1
        if conversation.expected.product_transitions:
            product_switch_conversations += 1
    return {
        "rag_turns_observed": rag_turns,
        "lead_conversations": lead_conversations,
        "support_conversations": support_conversations,
        "product_switch_conversations": product_switch_conversations,
    }
