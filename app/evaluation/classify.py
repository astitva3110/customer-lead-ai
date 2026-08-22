"""Observe traces and emit candidate failures only when verified GT is violated."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.catalog import CatalogItem
from app.evaluation.ground_truth import (
    SOURCE_CATALOG,
    SOURCE_RULE,
    evidence_ranks,
    expected_blocked,
    grounded_ok,
    has_evidence,
    is_insufficient,
    named_product,
    norm_product,
    rewrite_required,
    verified_answerable,
    verified_corpus_gap,
    verified_items,
)
from app.evaluation.schema import GeneratedConversation, TurnMessage

FAILURE_TYPES = (
    "NONE",
    "GUARDRAIL_FAILURE",
    "QUERY_REWRITE_FAILURE",
    "RETRIEVAL_FAILURE",
    "RERANKER_FAILURE",
    "FINAL_CONTEXT_FAILURE",
    "GENERATION_FAILURE",
    "GROUNDING_FAILURE",
    "STATE_FAILURE",
    "TOOL_FAILURE",
    "CORPUS_GAP",
)

LAYER_ORDER = (
    "guardrail",
    "query_rewrite",
    "retrieval",
    "reranker",
    "final_context",
    "generation",
    "grounding",
    "state",
    "tools",
)

_COMPARISON_RE = re.compile(r"\bdifference\b|\bcompare\b|\bvs\.?\b", re.IGNORECASE)


@dataclass
class TurnClassification:
    conversation_id: str
    turn: int
    category: str
    failure_type: str
    first_failing_layer: str | None
    passed: bool
    pattern: str
    product: str | None = None
    knowledge_keys: list[str] = field(default_factory=list)
    trace_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    ground_truth_source: str | None = None
    observations: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate(self) -> bool:
        return not self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn": self.turn,
            "category": self.category,
            "failure_type": self.failure_type,
            "first_failing_layer": self.first_failing_layer,
            "passed": self.passed,
            "candidate": self.candidate,
            "pattern": self.pattern,
            "product": self.product,
            "knowledge_keys": list(self.knowledge_keys),
            "trace_id": self.trace_id,
            "details": dict(self.details),
            "ground_truth_source": self.ground_truth_source,
            "observations": dict(self.observations),
        }


def classify_turn(
    conversation: GeneratedConversation,
    message: TurnMessage,
    trace: dict[str, Any] | None,
    *,
    catalog: dict[str, CatalogItem],
    prior_tools: list[str],
    error: str | None = None,
) -> TurnClassification:
    trace = trace or {}
    trace_id = str(trace.get("trace_id") or (trace.get("request") or {}).get("trace_id") or "")
    after = trace.get("state_after") or {}
    before = trace.get("state_before") or {}
    turn_info = trace.get("turn_understanding") or {}
    retrieval = trace.get("retrieval") or {}
    grounding = trace.get("grounding") or {}
    generation = trace.get("generation") or {}
    tool = trace.get("tool_execution") or {}
    query = trace.get("query") or {}
    response_text = str((trace.get("response") or {}).get("final_response") or "")
    guardrail = trace.get("guardrail") or {}
    blocked = bool(guardrail.get("blocked")) or (trace.get("response") or {}).get("final_mode") == "REJECTED"
    items = verified_items(message.knowledge_keys, catalog)
    answerable = verified_answerable(items)
    gaps = verified_corpus_gap(items)
    named = named_product(message.text)
    product_before = before.get("current_product")
    product_after = after.get("current_product")
    observations = {
        "blocked": blocked,
        "turn_intent": turn_info.get("turn_intent") or after.get("current_turn_intent"),
        "needs_rag": turn_info.get("needs_rag"),
        "retrieval_used": bool(retrieval.get("candidates") or retrieval.get("raw_count")),
        "rewrite_executed": bool(query.get("rewrite_executed")),
        "grounded": grounding.get("grounded_returned"),
        "validator_reason": grounding.get("validator_reason") or generation.get("validator_reason"),
        "tool_name": tool.get("tool_name"),
        "tool_executed": bool(tool.get("tool_executed")),
        "lead_status": after.get("lead_status"),
        "support_status": after.get("support_status"),
        "product_before": product_before,
        "product_after": product_after,
        "named_product": named or None,
        "verified_knowledge_keys": [item.knowledge_key for item in items],
        "scenario_hypothesis": {
            "expected_intent": message.expected_intent,
            "expected_lead_created": conversation.expected.expected_lead_created,
            "expected_ticket_created": conversation.expected.expected_ticket_created,
            "needs_rag": message.needs_rag,
            "expected_blocked": message.expected_blocked,
        },
    }
    verified_keys = [item.knowledge_key for item in items]
    base = dict(
        conversation_id=conversation.conversation_id,
        turn=message.turn,
        category=conversation.category,
        product=named or product_after or message.expected_product,
        knowledge_keys=verified_keys,
        trace_id=trace_id,
        observations=observations,
    )
    if error:
        return TurnClassification(
            failure_type="TOOL_FAILURE",
            first_failing_layer="tools",
            passed=False,
            pattern="TURN_ERROR",
            details={"error": error},
            ground_truth_source=SOURCE_RULE,
            **base,
        )

    required_block = expected_blocked(message.text)
    if required_block:
        if blocked:
            return TurnClassification(
                failure_type="NONE",
                first_failing_layer=None,
                passed=True,
                pattern="GUARDRAIL_BLOCKED",
                ground_truth_source=SOURCE_RULE,
                **base,
            )
        return TurnClassification(
            failure_type="GUARDRAIL_FAILURE",
            first_failing_layer="guardrail",
            passed=False,
            pattern="GUARDRAIL_MISSED_INJECTION",
            details={"rule": required_block, "blocked": blocked},
            ground_truth_source=SOURCE_RULE,
            **base,
        )
    if blocked:
        return TurnClassification(
            failure_type="GUARDRAIL_FAILURE",
            first_failing_layer="guardrail",
            passed=False,
            pattern="GUARDRAIL_FALSE_POSITIVE",
            ground_truth_source=SOURCE_RULE,
            **base,
        )

    raw_rows = list(retrieval.get("candidates") or [])
    rerank_rows = list((trace.get("reranking") or {}).get("candidates") or [])
    final_rows = list((trace.get("final_context") or {}).get("chunks") or [])
    retrieved = bool(raw_rows or final_rows or retrieval.get("raw_count"))
    rag_path = "query" in trace or "retrieval" in trace or retrieved

    if rewrite_required(message.text, product_before) and rag_path:
        rewritten = str(query.get("rewritten_query") or "")
        if not query.get("rewrite_executed") or (
            product_before and str(product_before).lower() not in rewritten.lower()
        ):
            return TurnClassification(
                failure_type="QUERY_REWRITE_FAILURE",
                first_failing_layer="query_rewrite",
                passed=False,
                pattern="PRODUCT_NOT_BOUND",
                details={"rewritten_query": rewritten, "product": product_before},
                ground_truth_source=SOURCE_RULE,
                **base,
            )

    if answerable:
        if not retrieved:
            return TurnClassification(
                failure_type="RETRIEVAL_FAILURE",
                first_failing_layer="retrieval",
                passed=False,
                pattern="EMPTY_RETRIEVAL",
                ground_truth_source=SOURCE_CATALOG,
                **base,
            )
        if not has_evidence(raw_rows, answerable):
            return TurnClassification(
                failure_type="RETRIEVAL_FAILURE",
                first_failing_layer="retrieval",
                passed=False,
                pattern="EXPECTED_NOT_IN_RAW",
                details={"raw_count": len(raw_rows)},
                ground_truth_source=SOURCE_CATALOG,
                **base,
            )
        rerank_enabled = bool((trace.get("reranking") or {}).get("enabled"))
        in_final = has_evidence(final_rows, answerable)
        if not in_final and rerank_enabled:
            raw_rank = (evidence_ranks(raw_rows, answerable) or [None])[0]
            rerank_rank = (evidence_ranks(rerank_rows, answerable) or [None])[0]
            dropped = rerank_rank is None or (
                raw_rank is not None and rerank_rank > max(raw_rank, 5) and not in_final
            )
            if dropped:
                return TurnClassification(
                    failure_type="RERANKER_FAILURE",
                    first_failing_layer="reranker",
                    passed=False,
                    pattern="RERANKER_DROPPED_EVIDENCE",
                    details={"raw_rank": raw_rank, "reranked_rank": rerank_rank},
                    ground_truth_source=SOURCE_CATALOG,
                    **base,
                )
        if not in_final:
            return TurnClassification(
                failure_type="FINAL_CONTEXT_FAILURE",
                first_failing_layer="final_context",
                passed=False,
                pattern="EVIDENCE_NOT_IN_FINAL_CONTEXT",
                ground_truth_source=SOURCE_CATALOG,
                **base,
            )
        validator_reason = str(grounding.get("validator_reason") or generation.get("validator_reason") or "")
        if validator_reason in {
            "invalid_json",
            "missing_fields",
            "invalid_grounded_flag",
            "invalid_answer",
            "invalid_source_ids",
        }:
            return TurnClassification(
                failure_type="GENERATION_FAILURE",
                first_failing_layer="generation",
                passed=False,
                pattern="MALFORMED_GENERATION",
                details={"validator_reason": validator_reason},
                ground_truth_source=SOURCE_RULE,
                **base,
            )
        if validator_reason in {"source_id_not_found", "empty_source_ids"}:
            return TurnClassification(
                failure_type="GROUNDING_FAILURE",
                first_failing_layer="grounding",
                passed=False,
                pattern="INVALID_SOURCE_IDS",
                details={"validator_reason": validator_reason},
                ground_truth_source=SOURCE_RULE,
                **base,
            )
        if is_insufficient(response_text, grounding):
            return TurnClassification(
                failure_type="GENERATION_FAILURE",
                first_failing_layer="generation",
                passed=False,
                pattern="GROUNDED_CONTEXT_UNUSED",
                ground_truth_source=SOURCE_CATALOG,
                **base,
            )
        if grounding.get("invalid_source_ids"):
            return TurnClassification(
                failure_type="GROUNDING_FAILURE",
                first_failing_layer="grounding",
                passed=False,
                pattern="SOURCE_ID_NOT_IN_CONTEXT",
                ground_truth_source=SOURCE_RULE,
                **base,
            )

    if gaps:
        if retrieved and not is_insufficient(response_text, grounding) and grounded_ok(grounding):
            return TurnClassification(
                failure_type="GROUNDING_FAILURE",
                first_failing_layer="grounding",
                passed=False,
                pattern="CORPUS_GAP_HALLUCINATION",
                details={"grounded": grounding.get("grounded_returned")},
                ground_truth_source=SOURCE_CATALOG,
                **base,
            )
        return TurnClassification(
            failure_type="CORPUS_GAP",
            first_failing_layer=None,
            passed=True,
            pattern="CORPUS_GAP_OBSERVED",
            ground_truth_source=SOURCE_CATALOG,
            **base,
        )

    if named and not _comparison_turn(message.text):
        if norm_product(product_after) != norm_product(named):
            return TurnClassification(
                failure_type="STATE_FAILURE",
                first_failing_layer="state",
                passed=False,
                pattern="STATE_FAILURE_PRODUCT_SWITCH",
                details={"from": product_before, "to": named, "observed": product_after},
                ground_truth_source=SOURCE_RULE,
                **base,
            )

    tool_name = str(tool.get("tool_name") or "")
    if tool.get("tool_executed") and tool.get("success") is False:
        return TurnClassification(
            failure_type="TOOL_FAILURE",
            first_failing_layer="tools",
            passed=False,
            pattern="TOOL_EXECUTION_FAILED",
            details={"tool_name": tool_name},
            ground_truth_source=SOURCE_RULE,
            **base,
        )
    if tool.get("tool_executed") and tool_name and tool_name in prior_tools:
        return TurnClassification(
            failure_type="TOOL_FAILURE",
            first_failing_layer="tools",
            passed=False,
            pattern="DUPLICATE_TOOL_EXECUTION",
            details={"tool_name": tool_name},
            ground_truth_source=SOURCE_RULE,
            **base,
        )

    return TurnClassification(
        failure_type="NONE",
        first_failing_layer=None,
        passed=True,
        pattern="OBSERVED",
        **base,
    )


def classify_conversation(
    conversation: GeneratedConversation,
    turn_results: list[TurnClassification],
    last_trace: dict[str, Any] | None,
) -> TurnClassification:
    del last_trace
    for item in turn_results:
        if item.candidate:
            return item
    last = turn_results[-1] if turn_results else TurnClassification(
        conversation_id=conversation.conversation_id,
        turn=0,
        category=conversation.category,
        failure_type="NONE",
        first_failing_layer=None,
        passed=True,
        pattern="OBSERVED",
    )
    return TurnClassification(
        conversation_id=conversation.conversation_id,
        turn=last.turn,
        category=conversation.category,
        failure_type="NONE",
        first_failing_layer=None,
        passed=True,
        pattern="OBSERVED",
        product=last.product,
        knowledge_keys=list(last.knowledge_keys),
        trace_id=last.trace_id,
        observations=dict(last.observations),
    )


def _comparison_turn(message: str) -> bool:
    return bool(_COMPARISON_RE.search(message or ""))
