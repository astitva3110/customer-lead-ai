from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from app.services.diagnostics.models import ChatTrace
from app.services.diagnostics.redact import redact_mapping, redact_text

NOT_AVAILABLE = "NOT_AVAILABLE"
SECTION = "------------------------------------------------------------"
BANNER = "============================================================"
FIRST_FAILURES = (
    "NONE",
    "GUARDRAIL",
    "ROUTER",
    "QUERY_REWRITE",
    "RETRIEVAL",
    "RERANKER",
    "FINAL_CONTEXT",
    "GENERATION",
    "GROUNDING",
    "RESPONSE",
)


def print_chat_debug_console(trace: ChatTrace | dict[str, Any], stream: TextIO | None = None) -> None:
    text = format_chat_debug_console(trace)
    if not text.endswith("\n"):
        text += "\n"
    output = stream or sys.stdout
    output.write(text)
    output.flush()


def format_chat_debug_console(trace: ChatTrace | dict[str, Any]) -> str:
    payload = _payload(trace)
    lines: list[str] = []
    lines.extend(_header(payload))
    lines.extend(_section_guardrail(payload))
    lines.extend(_section_router(payload))
    lines.extend(_section_llamaindex(payload))
    lines.extend(_section_raw_retrieval(payload))
    lines.extend(_section_hybrid(payload))
    lines.extend(_section_rerank_input(payload))
    lines.extend(_section_rerank_output(payload))
    lines.extend(_section_final_context(payload))
    lines.extend(_section_generation(payload))
    lines.extend(_section_raw_llm(payload))
    lines.extend(_section_grounding(payload))
    lines.extend(_section_response(payload))
    lines.extend(_section_state_after(payload))
    lines.extend(_section_latency(payload))
    lines.extend(_section_diagnostic_summary(payload))
    return "\n".join(lines) + "\n"


def classify_first_failure(trace: ChatTrace | dict[str, Any]) -> str:
    payload = _payload(trace)
    guard = payload.get("guardrail") or {}
    if guard.get("blocked"):
        return "GUARDRAIL"
    turn = payload.get("turn_understanding") or {}
    query = payload.get("query") or {}
    retrieval = payload.get("retrieval") or {}
    rerank = payload.get("reranking") or {}
    context = payload.get("final_context") or {}
    generation = payload.get("generation") or {}
    grounding = payload.get("grounding") or {}
    response = payload.get("response") or {}
    needs_rag = bool(turn.get("needs_rag"))
    if (
        str(turn.get("turn_intent") or "") == "KNOWLEDGE"
        and turn.get("needs_rag") is False
    ):
        return "ROUTER"
    if query.get("rewrite_executed") and not (query.get("rewritten_query") or "").strip():
        return "QUERY_REWRITE"
    raw = _raw_candidates(payload)
    raw_ids = _chunk_ids(raw)
    reranked = list((rerank.get("candidates") or []) if rerank.get("enabled") else [])
    rerank_ids = _chunk_ids(reranked)
    final_ids = _chunk_ids(context.get("chunks") or [])
    if needs_rag and not raw_ids:
        return "RETRIEVAL"
    if needs_rag and raw_ids and rerank.get("enabled") and rerank_ids:
        top_raw = raw_ids[0]
        kept = set(rerank_ids[: max(1, len(final_ids) or len(rerank_ids))])
        if top_raw not in kept and top_raw not in set(final_ids):
            return "RERANKER"
    if needs_rag and raw_ids and not final_ids:
        return "FINAL_CONTEXT"
    reason = str(grounding.get("validator_reason") or "")
    if reason in {"invalid_json", "missing_fields", "invalid_grounded_flag", "invalid_answer"}:
        return "GENERATION"
    if needs_rag and not (generation.get("raw_output") or generation.get("generation_result")):
        if final_ids:
            return "GENERATION"
    if grounding.get("validator_result") is False and reason not in {"", "empty_hits"}:
        return "GROUNDING"
    if needs_rag and reason == "empty_hits":
        return "RETRIEVAL"
    if not str(response.get("final_response") or "").strip():
        return "RESPONSE"
    return "NONE"


def diagnostic_summary(trace: ChatTrace | dict[str, Any]) -> dict[str, Any]:
    payload = _payload(trace)
    retrieval = payload.get("retrieval") or {}
    rerank = payload.get("reranking") or {}
    context = payload.get("final_context") or {}
    generation = payload.get("generation") or {}
    grounding = payload.get("grounding") or {}
    response = payload.get("response") or {}
    after = payload.get("state_after") or {}
    raw = _raw_candidates(payload)
    reranked = list(rerank.get("candidates") or [])
    result = generation.get("generation_result") or {}
    expected = after.get("current_product") or NOT_AVAILABLE
    context_has_expected = _expected_in_context(expected, context.get("chunks") or [])
    validator_pass = grounding.get("validator_result")
    grounding_label = "PASS" if validator_pass else ("FAIL" if grounding else NOT_AVAILABLE)
    return {
        "retrieval": "FOUND" if raw else "NOT_FOUND",
        "expected_product_section": expected,
        "best_raw_match": _match_label(raw[0] if raw else None),
        "best_reranked_match": _match_label(reranked[0] if reranked else None),
        "expected_information_in_final_context": context_has_expected,
        "llm_generated_answer": "YES" if (result.get("answer") or generation.get("raw_output")) else "NO",
        "llm_source_ids": result.get("source_ids") if result else grounding.get("returned_source_ids"),
        "llm_grounded": result.get("grounded") if result.get("grounded") is not None else grounding.get("grounded_returned"),
        "grounding": grounding_label,
        "grounding_failure": grounding.get("validator_reason") if not validator_pass else "",
        "grounding_result": grounding.get("validator_reason") or grounding_label,
        "final_response": response.get("final_response") or "",
        "first_failure": classify_first_failure(payload),
        "raw_top_match": _match_label(raw[0] if raw else None),
        "reranked_top_match": _match_label(reranked[0] if reranked else None),
        "final_context_contains_expected": context_has_expected,
        "trace_id": payload.get("trace_id") or (payload.get("request") or {}).get("trace_id") or "",
        "candidate_count": retrieval.get("raw_count") or len(raw),
    }


def _payload(trace: ChatTrace | dict[str, Any]) -> dict[str, Any]:
    if isinstance(trace, ChatTrace):
        return trace.to_dict()
    return dict(trace or {})


def _header(payload: dict[str, Any]) -> list[str]:
    request = payload.get("request") or {}
    return [
        BANNER,
        "CHAT DEBUG",
        BANNER,
        "",
        f"TRACE ID: {payload.get('trace_id') or request.get('trace_id') or ''}",
        f"CONVERSATION ID: {request.get('conversation_id') or ''}",
        f"CHANNEL: {request.get('channel') or ''}",
        f"ORIGIN: {request.get('origin') or ''}",
        "USER MESSAGE:",
        redact_text(str(request.get("user_message") or "")),
        "",
    ]


def _section_guardrail(payload: dict[str, Any]) -> list[str]:
    guard = payload.get("guardrail") or {}
    executed = guard.get("executed")
    blocked = bool(guard.get("blocked"))
    status = "executed" if executed else "skipped"
    decision = "block" if blocked else "allow"
    return [
        SECTION,
        "1. GUARDRAIL",
        SECTION,
        "",
        f"status: {status}",
        f"decision: {decision}",
        f"reason: {_value(guard.get('reason'))}",
        f"latency: {_value(guard.get('latency_ms'))}",
        "",
    ]


def _section_router(payload: dict[str, Any]) -> list[str]:
    turn = payload.get("turn_understanding") or {}
    query = payload.get("query") or {}
    after = payload.get("state_after") or {}
    before = payload.get("state_before") or {}
    user_context = redact_mapping(turn.get("user_context") or after.get("user_context") or {})
    return [
        SECTION,
        "2. CONVERSATION / ROUTER",
        SECTION,
        "",
        f"conversation_goal: {after.get('conversation_goal') or before.get('conversation_goal') or ''}",
        f"current_turn_intent: {turn.get('turn_intent') or after.get('current_turn_intent') or ''}",
        f"needs_rag: {turn.get('needs_rag')}",
        f"lead_intent: {turn.get('lead_intent')}",
        f"support_intent: {turn.get('support_intent')}",
        f"product: {after.get('current_product') or before.get('current_product') or ''}",
        f"user_context: {user_context}",
        f"rewritten_query: {query.get('rewritten_query') or ''}",
        f"semantic_router_used: {(payload.get('semantic_router') or {}).get('used')}",
        f"semantic_router_route: {(payload.get('semantic_router') or {}).get('route') or ''}",
        f"semantic_router_canonical_query: {(payload.get('semantic_router') or {}).get('canonical_query') or ''}",
        f"semantic_router_confidence: {(payload.get('semantic_router') or {}).get('confidence')}",
        f"semantic_router_model: {(payload.get('semantic_router') or {}).get('model') or ''}",
        f"semantic_router_tokens: {(payload.get('semantic_router') or {}).get('total_tokens')}",
        f"semantic_router_latency_ms: {(payload.get('semantic_router') or {}).get('latency_ms')}",
        "",
    ]


def _section_llamaindex(payload: dict[str, Any]) -> list[str]:
    llama = payload.get("llamaindex") or {}
    retrieval = payload.get("retrieval") or {}
    return [
        SECTION,
        "3. LLAMAINDEX",
        SECTION,
        "",
        f"llamaindex_enabled: {llama.get('enabled', False)}",
        f"index/service used: {llama.get('service') or NOT_AVAILABLE}",
        f"retriever used: {llama.get('retriever') or llama.get('adapter') or NOT_AVAILABLE}",
        f"retrieval_query: {llama.get('retrieval_query') or retrieval.get('retrieval_query') or ''}",
        f"retrieval_top_k: {_value(llama.get('retrieval_top_k') or retrieval.get('retrieval_k'))}",
        f"embedding_model: {_value(llama.get('embedding_model') or retrieval.get('embedding_model'))}",
        f"embedding_dimension: {_value(llama.get('embedding_dimension') or retrieval.get('embedding_dimension'))}",
        "",
    ]


def _section_raw_retrieval(payload: dict[str, Any]) -> list[str]:
    retrieval = payload.get("retrieval") or {}
    raw = _raw_candidates(payload)
    lines = [
        SECTION,
        "4. RAW RETRIEVAL",
        SECTION,
        "",
    ]
    if not raw:
        lines.append("No pre-rerank candidates captured.")
        lines.append("")
    for item in raw:
        lines.extend(_raw_block(item, prefix="RAW"))
    scores = [item.get("score") for item in raw if isinstance(item.get("score"), (int, float))]
    top = retrieval.get("raw_top_score")
    if top is None and scores:
        top = max(scores)
    low = retrieval.get("raw_score_min")
    high = retrieval.get("raw_score_max")
    if scores and low is None:
        low = min(scores)
        high = max(scores)
    score_range = NOT_AVAILABLE
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        score_range = f"{low:.4f} .. {high:.4f}"
    lines.extend(
        [
            f"RAW RETRIEVAL COUNT: {retrieval.get('raw_count') if retrieval.get('raw_count') is not None else len(raw)}",
            f"RAW TOP SCORE: {_score(top)}",
            f"RAW SCORE RANGE: {score_range}",
            "",
        ]
    )
    return lines


def _section_hybrid(payload: dict[str, Any]) -> list[str]:
    retrieval = payload.get("retrieval") or {}
    vector = list(retrieval.get("vector") or [])
    keyword = list(retrieval.get("keyword") or [])
    merged = list(retrieval.get("merged") or [])
    lines = [
        SECTION,
        "5. HYBRID RETRIEVAL BREAKDOWN",
        SECTION,
        "",
        "VECTOR RESULTS",
    ]
    lines.extend(_hybrid_lines(vector))
    lines.extend(["", "LEXICAL/BM25/FTS RESULTS"])
    lines.extend(_hybrid_lines(keyword))
    lines.extend(["", "MERGED RESULTS"])
    lines.extend(_hybrid_lines(merged))
    lines.append("")
    return lines


def _section_rerank_input(payload: dict[str, Any]) -> list[str]:
    rerank = payload.get("reranking") or {}
    enabled = bool(rerank.get("enabled"))
    incoming = list(rerank.get("input") or [])
    if not incoming:
        incoming = list((payload.get("retrieval") or {}).get("merged") or [])
    lines = [
        SECTION,
        "6. RERANKER INPUT",
        SECTION,
        "",
        f"reranker_model: {_value(rerank.get('model') or rerank.get('name'))}",
        f"reranker_enabled: {enabled}",
        f"candidate_count: {_value(rerank.get('input_count') if rerank.get('input_count') is not None else len(incoming))}",
        "",
    ]
    if not incoming:
        lines.append(NOT_AVAILABLE)
        lines.append("")
        return lines
    for item in incoming:
        lines.extend(
            [
                f"chunk_id: {item.get('chunk_id')}",
                f"vector_score: {_score(item.get('vector_score'))}",
                f"lexical_score: {_score(item.get('keyword_score'))}",
                f"combined_score: {_score(item.get('combined_score'))}",
                f"section: {_value(item.get('section_path'))}",
                "",
            ]
        )
    return lines


def _section_rerank_output(payload: dict[str, Any]) -> list[str]:
    rerank = payload.get("reranking") or {}
    lines = [
        SECTION,
        "7. RERANKER OUTPUT",
        SECTION,
        "",
    ]
    candidates = list(rerank.get("candidates") or [])
    if not rerank.get("enabled"):
        lines.append("reranker_enabled: False")
        lines.append("")
        return lines
    if not candidates:
        lines.append(NOT_AVAILABLE)
        lines.append("")
        return lines
    for item in candidates:
        lines.extend(
            [
                f"RERANKED #{item.get('rank')}",
                f"chunk_id: {item.get('chunk_id')}",
                f"score: {_score(item.get('rerank_score') if item.get('rerank_score') is not None else item.get('score'))}",
                f"original_retrieval_rank: {_value(item.get('original_retrieval_rank'))}",
                f"section: {_value(item.get('section_path'))}",
                "text:",
                str(item.get("text_preview") or item.get("text") or ""),
                "",
            ]
        )
    return lines


def _section_final_context(payload: dict[str, Any]) -> list[str]:
    context = payload.get("final_context") or {}
    chunks = list(context.get("chunks") or [])
    lines = [
        SECTION,
        "8. FINAL CONTEXT",
        SECTION,
        "",
    ]
    for item in chunks:
        lines.extend(
            [
                f"SOURCE_ID: {item.get('chunk_id')}",
                f"chunk_id: {item.get('chunk_id')}",
                f"section: {_value(item.get('section_path'))}",
                f"score: {_score(item.get('score'))}",
                "text:",
                str(item.get("text") or ""),
                "",
            ]
        )
    lines.extend([f"FINAL CONTEXT COUNT: {context.get('context_count') if context.get('context_count') is not None else len(chunks)}", ""])
    return lines


def _section_generation(payload: dict[str, Any]) -> list[str]:
    generation = payload.get("generation") or {}
    lines = [
        SECTION,
        "9. GENERATION",
        SECTION,
        "",
        f"provider: {_value(generation.get('provider'))}",
        f"model: {_value(generation.get('model'))}",
        f"temperature: {_value(generation.get('temperature'))}",
        f"max_tokens: {_value(generation.get('max_tokens'))}",
        f"generation_latency: {_value(generation.get('latency_ms'))}",
        f"llm_call_count: {_value(generation.get('llm_call_count'))}",
        f"parsed_grounded: {_value(generation.get('parsed_grounded'))}",
        f"parsed_source_ids: {_value(generation.get('parsed_source_ids'))}",
        f"valid_source_ids: {_value(generation.get('valid_source_ids'))}",
        f"invalid_source_ids: {_value(generation.get('invalid_source_ids'))}",
        f"validator_reason: {_value(generation.get('validator_reason'))}",
        "",
        "SYSTEM PROMPT",
        redact_text(str(generation.get("system_prompt") or NOT_AVAILABLE)),
        "",
        "USER PROMPT",
        redact_text(str(generation.get("user_prompt_body") or generation.get("user_prompt") or generation.get("prompt") or NOT_AVAILABLE)),
        "",
        "KNOWLEDGE CONTEXT",
        redact_text(str(generation.get("knowledge_context") or NOT_AVAILABLE)),
        "",
    ]
    return lines


def _section_raw_llm(payload: dict[str, Any]) -> list[str]:
    generation = payload.get("generation") or {}
    raw = generation.get("raw_model_output")
    if raw is None:
        raw = generation.get("raw_output")
    if raw is None:
        rendered = NOT_AVAILABLE
    else:
        rendered = redact_text(str(raw))
    return [
        SECTION,
        "10. RAW LLM OUTPUT",
        SECTION,
        "",
        rendered,
        "",
    ]


def _section_grounding(payload: dict[str, Any]) -> list[str]:
    grounding = payload.get("grounding") or {}
    generation = payload.get("generation") or {}
    result = generation.get("generation_result") or {}
    grounded = result.get("grounded")
    if grounded is None:
        grounded = grounding.get("grounded_returned")
    validator = grounding.get("validator_result")
    validator_label = "PASS" if validator else ("FAIL" if grounding else NOT_AVAILABLE)
    reason = grounding.get("validator_reason") or ""
    lines = [
        SECTION,
        "11. GROUNDING VALIDATOR",
        SECTION,
        "",
        f"grounded: {grounded}",
        f"validator_result: {validator_label}",
        f"validator_reason: {_value(reason)}",
        f"returned_source_ids: {grounding.get('returned_source_ids') or []}",
        f"valid_source_ids: {grounding.get('valid_source_ids') or []}",
        f"invalid_source_ids: {grounding.get('invalid_source_ids') or []}",
        "",
    ]
    if validator is False and reason:
        lines.extend([f"rejection_condition: {reason}", ""])
    return lines


def _section_response(payload: dict[str, Any]) -> list[str]:
    response = payload.get("response") or {}
    return [
        SECTION,
        "12. FINAL RESPONSE",
        SECTION,
        "",
        f"mode: {response.get('final_mode') or ''}",
        "response:",
        redact_text(str(response.get("final_response") or "")),
        "answer:",
        redact_text(str(response.get("final_response") or "")),
        f"sources: {response.get('source_ids') or []}",
        "",
    ]


def _section_state_after(payload: dict[str, Any]) -> list[str]:
    after = payload.get("state_after") or {}
    turn = payload.get("turn_understanding") or {}
    return [
        SECTION,
        "13. STATE AFTER",
        SECTION,
        "",
        f"conversation_goal: {after.get('conversation_goal') or ''}",
        f"current_turn_intent: {after.get('current_turn_intent') or turn.get('turn_intent') or ''}",
        f"product: {after.get('current_product') or ''}",
        f"lead_state: {after.get('lead_state') or after.get('lead_status') or ''}",
        f"support_state: {after.get('support_state') or after.get('support_status') or ''}",
        "",
    ]


def _section_latency(payload: dict[str, Any]) -> list[str]:
    latency = payload.get("latency") or {}
    return [
        SECTION,
        "14. LATENCY",
        SECTION,
        "",
        f"guardrail_ms: {_value(latency.get('guardrail_ms'))}",
        f"router_ms: {_value(latency.get('router_ms') if latency.get('router_ms') is not None else latency.get('routing_ms'))}",
        f"rewrite_ms: {_value(latency.get('rewrite_ms'))}",
        f"retrieval_ms: {_value(latency.get('retrieval_ms'))}",
        f"rerank_ms: {_value(latency.get('rerank_ms'))}",
        f"generation_ms: {_value(latency.get('generation_ms'))}",
        f"total_ms: {_value(latency.get('total_ms'))}",
        "",
    ]


def _section_diagnostic_summary(payload: dict[str, Any]) -> list[str]:
    summary = diagnostic_summary(payload)
    return [
        BANNER,
        "DIAGNOSTIC SUMMARY",
        BANNER,
        "",
        "RETRIEVAL:",
        str(summary["retrieval"]),
        "",
        "EXPECTED PRODUCT/SECTION:",
        str(summary["expected_product_section"]),
        "",
        "BEST RAW MATCH:",
        str(summary["best_raw_match"]),
        "",
        "BEST RERANKED MATCH:",
        str(summary["best_reranked_match"]),
        "",
        "EXPECTED INFORMATION IN FINAL CONTEXT:",
        str(summary["expected_information_in_final_context"]),
        "",
        "LLM GENERATED ANSWER:",
        str(summary["llm_generated_answer"]),
        "",
        "LLM SOURCE IDS:",
        str(summary["llm_source_ids"]),
        "",
        "GROUNDING:",
        str(summary["grounding"]),
        "",
        "GROUNDING FAILURE:",
        str(summary["grounding_failure"] or ""),
        "",
        "FINAL RESPONSE:",
        redact_text(str(summary["final_response"] or "")),
        "",
        "FIRST_FAILURE:",
        str(summary["first_failure"]),
        "",
    ]


def _raw_block(item: dict[str, Any], *, prefix: str) -> list[str]:
    section = item.get("section_path") or item.get("section") or ""
    return [
        f"[{prefix} #{item.get('rank')}]",
        f"chunk_id: {item.get('chunk_id')}",
        f"document_id: {item.get('document_id')}",
        f"score: {_score(item.get('score'))}",
        f"section: {section}",
        f"tokens: {_value(item.get('token_count'))}",
        f"type: {_value(item.get('chunk_type'))}",
        f"title: {_value(item.get('title'))}",
        "",
        "TEXT:",
        str(item.get("text_preview") or item.get("text") or ""),
        "",
    ]


def _hybrid_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [NOT_AVAILABLE]
    lines: list[str] = []
    for item in rows:
        lines.extend(
            [
                f"chunk_id: {item.get('chunk_id')}",
                f"vector_score: {_score(item.get('vector_score'))}",
                f"lexical_score: {_score(item.get('keyword_score'))}",
                f"combined_score: {_score(item.get('combined_score'))}",
            ]
        )
    return lines


def _raw_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = payload.get("retrieval") or {}
    if "candidates" in retrieval:
        return list(retrieval.get("candidates") or [])
    return list(retrieval.get("merged") or retrieval.get("vector") or [])


def _chunk_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("chunk_id")) for item in rows if item.get("chunk_id")]


def _expected_in_context(expected: Any, chunks: list[dict[str, Any]]) -> str:
    product = str(expected or "").strip()
    if not product or product == NOT_AVAILABLE:
        return "UNKNOWN"
    haystack = " ".join(
        f"{item.get('text') or ''} {item.get('section_path') or ''} {item.get('title') or ''}"
        for item in chunks
    ).lower()
    if not haystack:
        return "NO"
    return "YES" if product.lower() in haystack else "NO"


def _match_label(item: dict[str, Any] | None) -> str:
    if not item:
        return NOT_AVAILABLE
    section = item.get("section_path") or item.get("section") or ""
    return f"{item.get('chunk_id')} score={_score(item.get('score'))} section={section}"


def _score(value: Any) -> str:
    if value is None or value == "":
        return NOT_AVAILABLE
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _value(value: Any) -> str:
    if value is None or value == "":
        return NOT_AVAILABLE
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)
