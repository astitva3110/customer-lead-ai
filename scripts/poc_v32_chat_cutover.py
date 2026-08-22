#!/usr/bin/env python3
"""POC /chat cutover checks against chunk_embeddings_v3_2. Does not alter prompts or routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.ingestion.indexing import Phase12VectorStore

TRACE_DIR = ROOT / "reports/mass_eval/poc_v32_chat"
KNOWLEDGE_TURNS = [
    "Hi, what is TINY?",
    "What is BTE?",
    "What are the features of Bluup?",
    "What is the warranty of Bluup?",
    "Why should I buy from Earkart?",
    "Where is the office of Earkart?",
]
FLOW_TURNS = [
    "Hi, what is TINY?",
    "What is its warranty?",
    "How much does it cost?",
    "I want to buy it.",
    "Actually, tell me about BTE first.",
    "Okay, contact me.",
]


def section_label(path) -> str:
    if isinstance(path, list):
        return " / ".join(str(part) for part in path)
    return str(path or "")


def preview(text: str, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def load_trace(trace_id: str | None) -> dict:
    if not trace_id:
        return {}
    path = TRACE_DIR / f"{trace_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_rows(rows: list) -> list[dict]:
    out = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "section": section_label(row.get("section_path") or row.get("section")),
                "score": row.get("score"),
                "preview": preview(str(row.get("text") or row.get("text_preview") or "")),
            }
        )
    return out


def summarize_turn(payload: dict, trace: dict) -> dict:
    query = trace.get("query") or {}
    retrieval = trace.get("retrieval") or {}
    turn = trace.get("turn_understanding") or {}
    after = trace.get("state_after") or {}
    final_chunks = ((trace.get("final_context") or {}).get("chunks")) or []
    return {
        "conversation_id": payload.get("conversation_id"),
        "mode": payload.get("mode") or (trace.get("response") or {}).get("final_mode"),
        "message": (trace.get("request") or {}).get("message") or payload.get("message") or "",
        "rewritten_query": query.get("rewritten_query") or retrieval.get("retrieval_query"),
        "rewrite_executed": query.get("rewrite_executed"),
        "product": after.get("current_product"),
        "intent": turn.get("turn_intent"),
        "needs_rag": turn.get("needs_rag"),
        "lead_status": after.get("lead_status"),
        "awaiting_field": after.get("awaiting_field"),
        "tool_executed": (trace.get("tool_execution") or {}).get("tool_executed"),
        "vector_candidate_count": retrieval.get("vector_candidate_count"),
        "response": payload.get("response") or payload.get("answer") or "",
        "vector_top": candidate_rows(retrieval.get("vector") or []),
        "final_sections": [
            section_label(chunk.get("section_path")) + " :: " + preview(str(chunk.get("text") or ""))
            for chunk in final_chunks[:4]
            if isinstance(chunk, dict)
        ],
        "debug_trace_id": payload.get("debug_trace_id"),
        "trace_vector_table_diagnostic": retrieval.get("vector_table"),
        "retrieval_version": retrieval.get("corpus_version"),
    }


def post_turn(client, conversation_id: str | None, message: str) -> dict:
    body = {"message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    response = client.post("/chat", json=body)
    try:
        payload = response.json()
    except Exception:
        payload = {"response": response.text, "error": "non_json"}
    if not isinstance(payload, dict):
        payload = {"response": str(payload), "error": "unexpected_payload"}
    payload["_http_status"] = response.status_code
    payload["message"] = message
    return payload


def run_suite(client, *, conversation_id: str, messages: list[str]) -> list[dict]:
    rows = []
    current_id = conversation_id
    for index, message in enumerate(messages):
        payload = post_turn(client, current_id if index == 0 else current_id, message)
        if payload.get("conversation_id"):
            current_id = payload["conversation_id"]
        trace = load_trace(payload.get("debug_trace_id"))
        row = summarize_turn(payload, trace)
        row["message"] = message
        row["_http_status"] = payload.get("_http_status")
        if payload.get("error"):
            row["error"] = payload["error"]
        rows.append(row)
        print(
            f"[{current_id}] {message!r} -> mode={row['mode']} http={row['_http_status']}",
            flush=True,
        )
        print(f"    {preview(row['response'], 220)}", flush=True)
    return rows


def format_text(payload: dict) -> str:
    retrieval = payload["retrieval"]
    lines = [
        "POC_V32_CHAT_CUTOVER",
        "not_permanent: true",
        f"vector_table: {retrieval['vector_table']}",
        f"embedding_version: {retrieval['embedding_version']}",
        f"row_count: {payload.get('row_count')}",
        "pipeline_otherwise_unchanged: true",
        "note: embedding_version must match rows in chunk_embeddings_v3_2 or PGVector returns nothing.",
        "",
        "KNOWLEDGE_PROBES",
    ]
    for row in payload["knowledge"]:
        lines.append(f"  Q: {row['message']}")
        lines.append(
            f"     mode={row['mode']} intent={row.get('intent')} rewritten={row['rewritten_query']} product={row['product']}"
        )
        lines.append(f"     answer: {preview(row['response'], 320)}")
        if row.get("final_sections"):
            lines.append(f"     context: {row['final_sections'][0]}")
        if row.get("vector_top"):
            top = row["vector_top"][0]
            lines.append(f"     vector@1: {top['section']} :: {top['preview']}")
    lines.extend(["", "CONVERSATION_FLOW"])
    for index, row in enumerate(payload["flow"], start=1):
        lines.append(f"  {index}. {row['message']}")
        lines.append(
            f"     mode={row['mode']} intent={row.get('intent')} rewritten={row['rewritten_query']} product={row['product']}"
        )
        lines.append(
            f"     lead={row.get('lead_status')} awaiting={row.get('awaiting_field')} tool={row.get('tool_executed')}"
        )
        lines.append(f"     answer: {preview(row['response'], 320)}")
        if row.get("final_sections"):
            lines.append(f"     context: {row['final_sections'][0]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    config = RetrievalConfig.from_yaml(ROOT / "configs/retrieval/v2.yaml")
    if config.vector_table != "chunk_embeddings_v3_2":
        raise SystemExit(f"POC expected chunk_embeddings_v3_2, got {config.vector_table}")
    store = Phase12VectorStore(table_name=config.vector_table)
    row_count = store.count(embedding_version=config.embedding_version)
    if row_count <= 0:
        raise SystemExit(
            f"POC table {config.vector_table} has 0 rows for {config.embedding_version}"
        )
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    settings.chat_trace_enabled = True
    settings.chat_trace_include_full_context = True
    settings.chat_trace_output_dir = TRACE_DIR

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    knowledge_rows = []
    for index, message in enumerate(KNOWLEDGE_TURNS, start=1):
        knowledge_rows.extend(
            run_suite(client, conversation_id=f"poc-v32-k{index:02d}", messages=[message])
        )
    flow_rows = run_suite(client, conversation_id="poc-v32-flow", messages=FLOW_TURNS)
    payload = {
        "retrieval": config.to_dict(),
        "row_count": row_count,
        "knowledge": knowledge_rows,
        "flow": flow_rows,
    }
    out = ROOT / "reports/mass_eval"
    json_path = out / "poc_v32_chat.json"
    txt_path = out / "poc_v32_chat.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(format_text(payload), encoding="utf-8")
    print(format_text(payload))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
