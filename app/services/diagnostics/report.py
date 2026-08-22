from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.diagnostics.models import ChatTrace
from app.services.diagnostics.redact import redact_mapping


def write_chat_trace(trace: ChatTrace, output_dir: Path | None = None) -> tuple[Path, Path]:
    directory = Path(output_dir or settings.chat_trace_output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trace_id = trace.trace_id or "unknown"
    json_path = directory / f"{trace_id}.json"
    txt_path = directory / f"{trace_id}.txt"
    payload = redact_mapping(trace.to_dict())
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(format_chat_trace_text(payload), encoding="utf-8")
    return json_path, txt_path


def load_chat_trace(trace_id: str, output_dir: Path | None = None) -> dict[str, Any]:
    directory = Path(output_dir or settings.chat_trace_output_dir)
    path = directory / f"{trace_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def format_chat_trace_text(payload: dict[str, Any]) -> str:
    request = payload.get("request") or {}
    before = payload.get("state_before") or {}
    turn = payload.get("turn_understanding") or {}
    query = payload.get("query") or {}
    retrieval = payload.get("retrieval") or {}
    rerank = payload.get("reranking") or {}
    context = payload.get("final_context") or {}
    generation = payload.get("generation") or {}
    grounding = payload.get("grounding") or {}
    response = payload.get("response") or {}
    after = payload.get("state_after") or {}
    latency = payload.get("latency") or {}
    tool = payload.get("tool_execution") or {}
    lines = [
        "============================================================",
        "CHAT TRACE",
        "============================================================",
        "",
        "TRACE",
        f"id: {payload.get('trace_id') or request.get('trace_id') or ''}",
        "",
        "REQUEST",
        f"conversation_id: {request.get('conversation_id') or ''}",
        f"trace_id: {request.get('trace_id') or ''}",
        "message:",
        str(request.get("user_message") or ""),
        "",
        "STATE BEFORE",
        f"goal: {before.get('conversation_goal')}",
        f"product: {before.get('current_product')}",
        f"awaiting_field: {before.get('awaiting_field')}",
        f"previous_assistant_message: {before.get('previous_assistant_message')}",
        "",
        "GUARDRAIL",
        f"executed: {(payload.get('guardrail') or {}).get('executed')}",
        f"blocked: {(payload.get('guardrail') or {}).get('blocked')}",
        f"reason: {(payload.get('guardrail') or {}).get('reason')}",
        "",
        "TURN",
        f"intent: {turn.get('turn_intent')}",
        f"needs_rag: {turn.get('needs_rag')}",
        f"method: {turn.get('method')}",
        f"confidence: {turn.get('confidence')}",
        "",
        "QUERY",
        f"original: {query.get('original_user_query')}",
        f"normalized: {query.get('normalized_query')}",
        f"rewritten: {query.get('rewritten_query')}",
        f"rewrite_executed: {query.get('rewrite_executed')}",
        "",
        "RETRIEVAL",
        f"backend: {retrieval.get('backend')}",
        f"corpus: {retrieval.get('corpus_version')}",
        f"k: {retrieval.get('retrieval_k')}",
        f"latency: {retrieval.get('latency_ms')}",
    ]
    for item in retrieval.get("candidates") or []:
        lines.extend(
            [
                "",
                f"#{item.get('rank')} score={item.get('score')}",
                f"chunk={item.get('chunk_id')}",
                f"section={item.get('section_path')}",
                f"text={item.get('text_preview')}",
            ]
        )
    if rerank:
        lines.extend(
            [
                "",
                "RERANK",
                f"enabled: {rerank.get('enabled')}",
                f"name: {rerank.get('name')}",
                f"latency: {rerank.get('latency_ms')}",
            ]
        )
        for item in rerank.get("candidates") or []:
            lines.append(f"#{item.get('rank')} rerank={item.get('rerank_score') or item.get('score')} chunk={item.get('chunk_id')}")
    lines.extend(["", "FINAL CONTEXT", f"context_count: {context.get('context_count')}"])
    for item in context.get("chunks") or []:
        lines.extend(
            [
                "",
                f"[{item.get('position')}]",
                f"chunk_id: {item.get('chunk_id')}",
                f"title: {item.get('title')}",
                f"section: {item.get('section_path')}",
                "text:",
                str(item.get("text") or ""),
            ]
        )
    result = generation.get("generation_result") or {}
    lines.extend(
        [
            "",
            "GENERATION",
            f"provider: {generation.get('provider')}",
            f"model: {generation.get('model')}",
            f"temperature: {generation.get('temperature')}",
            f"latency: {generation.get('latency_ms')}",
            "",
            "GROUNDING",
            f"returned: {grounding.get('grounded_returned')}",
            f"validated: {grounding.get('validator_result')}",
            f"reason: {grounding.get('validator_reason')}",
            f"returned_source_ids: {grounding.get('returned_source_ids')}",
            f"available_source_ids: {grounding.get('available_source_ids')}",
            f"fallback: {grounding.get('fallback_triggered')}",
            "",
            "FINAL RESPONSE",
            str(response.get("final_response") or result.get("answer") or ""),
            "",
            "TOOL",
            f"tool_executed: {tool.get('tool_executed')}",
            f"tool_name: {tool.get('tool_name')}",
            f"success: {tool.get('success')}",
            "",
            "STATE AFTER",
            f"goal: {after.get('conversation_goal')}",
            f"product: {after.get('current_product')}",
            f"lead_status: {after.get('lead_status')}",
            f"ticket_status: {after.get('support_status')}",
            "",
            "LATENCY",
            f"guardrail: {latency.get('guardrail_ms')}",
            f"routing: {latency.get('routing_ms')}",
            f"rewrite: {latency.get('rewrite_ms')}",
            f"retrieval: {latency.get('retrieval_ms')}",
            f"rerank: {latency.get('rerank_ms')}",
            f"generation: {latency.get('generation_ms')}",
            f"tool: {latency.get('tool_ms')}",
            f"total: {latency.get('total_ms')}",
        ]
    )
    errors = payload.get("errors") or []
    if errors:
        lines.extend(["", "ERRORS"])
        for item in errors:
            lines.append(f"{item.get('component')}: {item.get('error_type')} {item.get('error_message')}")
    return "\n".join(str(line) for line in lines)
