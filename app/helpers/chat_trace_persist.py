"""Pure mapping from ChatTrace to relational row dicts."""

from __future__ import annotations

from typing import Any

from app.services.diagnostics.models import ChatTrace

LAYER_ORDER = ("vector", "keyword", "merged", "reranked", "final")

LAYER_KEYS = (
    ("vector", "vector"),
    ("keyword", "keyword"),
    ("merged", "merged"),
)


def chat_turn_fields(trace: ChatTrace) -> dict[str, Any]:
    request = trace.request or {}
    query = trace.query or {}
    turn = trace.turn_understanding or {}
    response = trace.response or {}
    tool = trace.tool_execution or {}
    latency = trace.latency or {}
    retrieval = trace.retrieval or {}
    rerank = trace.reranking or {}
    llama = trace.llamaindex or {}
    timings = dict(retrieval.get("timings") or {})
    rewritten = query.get("rewritten_query") or retrieval.get("retrieval_query") or llama.get("retrieval_query")
    return {
        "trace_id": trace.trace_id,
        "conversation_id": str(request.get("conversation_id") or ""),
        "user_message": str(request.get("user_message") or ""),
        "rewritten_query": rewritten or None,
        "intent": str(turn.get("turn_intent") or ""),
        "response": str(response.get("final_response") or ""),
        "tool_name": tool.get("tool_name") or None,
        "tool_success": tool.get("success"),
        "lead_id": tool.get("lead_id") or None,
        "ticket_id": tool.get("ticket_id") or None,
        "backend": retrieval.get("backend") or llama.get("adapter"),
        "reranker_name": rerank.get("name"),
        "guardrail_ms": _float(latency.get("guardrail_ms")),
        "routing_ms": _float(latency.get("routing_ms") if latency.get("routing_ms") is not None else latency.get("router_ms")),
        "rewrite_ms": _float(latency.get("rewrite_ms")),
        "retrieval_ms": _float(latency.get("retrieval_ms")),
        "generation_ms": _float(latency.get("generation_ms")),
        "tool_ms": _float(latency.get("tool_ms") if latency.get("tool_ms") is not None else tool.get("latency_ms")),
        "total_ms": _float(latency.get("total_ms")),
        "vector_ms": _float(latency.get("vector_ms") if latency.get("vector_ms") is not None else timings.get("vector_ms")),
        "keyword_ms": _float(latency.get("keyword_ms") if latency.get("keyword_ms") is not None else timings.get("keyword_ms")),
        "merge_ms": _float(latency.get("merge_ms") if latency.get("merge_ms") is not None else timings.get("merge_ms")),
        "rerank_ms": _float(latency.get("rerank_ms") if latency.get("rerank_ms") is not None else timings.get("rerank_ms")),
        "threshold_ms": _float(latency.get("threshold_ms") if latency.get("threshold_ms") is not None else timings.get("threshold_ms")),
    }


def retrieval_layer_hit_fields(trace: ChatTrace) -> list[dict[str, Any]]:
    retrieval = trace.retrieval or {}
    rerank = trace.reranking or {}
    final = trace.final_context or {}
    rows: list[dict[str, Any]] = []
    for source_key, layer in LAYER_KEYS:
        for item in retrieval.get(source_key) or []:
            rows.append(_hit(trace.trace_id, layer, item))
    for item in rerank.get("candidates") or []:
        rows.append(_hit(trace.trace_id, "reranked", item))
    for item in final.get("chunks") or []:
        row = _hit(trace.trace_id, "final", item)
        if item.get("position") is not None:
            row["rank"] = int(item.get("position") or row["rank"])
        rows.append(row)
    return rows


def _hit(trace_id: str, layer: str, item: dict[str, Any]) -> dict[str, Any]:
    section = item.get("section_path") if item.get("section_path") is not None else item.get("section")
    return {
        "trace_id": trace_id,
        "layer": layer,
        "rank": int(item.get("rank") or item.get("position") or 0),
        "chunk_id": str(item.get("chunk_id") or ""),
        "document_id": str(item.get("document_id") or ""),
        "title": str(item.get("title") or ""),
        "section_path": _section_text(section),
        "vector_score": _float(item.get("vector_score")),
        "keyword_score": _float(item.get("keyword_score")),
        "rerank_score": _float(item.get("rerank_score") if item.get("rerank_score") is not None else item.get("score")),
        "combined_score": _float(item.get("combined_score") if item.get("combined_score") is not None else item.get("score")),
        "original_retrieval_rank": _int(item.get("original_retrieval_rank")),
        "text_preview": str(item.get("text_preview") or item.get("text") or ""),
    }


def _section_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " > ".join(str(part) for part in value)
    return str(value)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def group_hits_by_layer(hits: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {layer: [] for layer in LAYER_ORDER}
    extra: dict[str, list[Any]] = {}
    for item in hits:
        layer = getattr(item, "layer", None)
        if layer is None and isinstance(item, dict):
            layer = item.get("layer")
        key = str(layer or "")
        if key in grouped:
            grouped[key].append(item)
        else:
            extra.setdefault(key, []).append(item)
    grouped.update(extra)
    return grouped


def isoformat_utc(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

