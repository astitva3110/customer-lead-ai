from __future__ import annotations

from io import StringIO
from pathlib import Path

from app.config import settings
from app.services.diagnostics.console import (
    classify_first_failure,
    diagnostic_summary,
    format_chat_debug_console,
    print_chat_debug_console,
)
from app.services.diagnostics.models import ChatTrace
from app.services.diagnostics.report import format_chat_trace_text
from tests.unit.conversation.fakes import make_orchestrator


REQUIRED_SECTIONS = [
    "CHAT DEBUG",
    "TRACE ID:",
    "CONVERSATION ID:",
    "USER MESSAGE:",
    "1. GUARDRAIL",
    "2. CONVERSATION / ROUTER",
    "3. LLAMAINDEX",
    "4. RAW RETRIEVAL",
    "5. HYBRID RETRIEVAL BREAKDOWN",
    "VECTOR RESULTS",
    "LEXICAL/BM25/FTS RESULTS",
    "MERGED RESULTS",
    "6. RERANKER INPUT",
    "7. RERANKER OUTPUT",
    "8. FINAL CONTEXT",
    "9. GENERATION",
    "SYSTEM PROMPT",
    "USER PROMPT",
    "KNOWLEDGE CONTEXT",
    "10. RAW LLM OUTPUT",
    "11. GROUNDING VALIDATOR",
    "12. FINAL RESPONSE",
    "13. STATE AFTER",
    "14. LATENCY",
    "DIAGNOSTIC SUMMARY",
    "FIRST_FAILURE:",
]


def _sample_payload() -> dict:
    return {
        "trace_id": "01TESTDEBUGCONSOLE00000000",
        "request": {
            "trace_id": "01TESTDEBUGCONSOLE00000000",
            "conversation_id": "debug-blue",
            "user_message": "what is Bluup? call +91 9876543210",
        },
        "state_before": {"conversation_goal": "NONE", "current_product": None},
        "guardrail": {"executed": True, "blocked": False, "reason": None, "latency_ms": 1.2},
        "turn_understanding": {
            "turn_intent": "KNOWLEDGE",
            "needs_rag": True,
            "lead_intent": False,
            "support_intent": False,
            "user_context": {"phone": "+919876543210", "city": "Noida"},
        },
        "query": {"rewritten_query": "what is Bluup"},
        "llamaindex": {
            "enabled": True,
            "service": "LlamaIndexKnowledgeService",
            "retriever": "HybridLlamaRetriever",
            "retrieval_query": "what is Bluup",
            "retrieval_top_k": 6,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dimension": 1024,
        },
        "retrieval": {
            "candidates": [
                {
                    "rank": 1,
                    "chunk_id": "chunk-bluup",
                    "document_id": "doc-1",
                    "score": 0.7421,
                    "section_path": "Products > Bluup",
                    "token_count": 54,
                    "chunk_type": "product_feature",
                    "title": "Bluup",
                    "text_preview": "Bluup is a hearing aid.",
                    "vector_score": 0.74,
                    "keyword_score": 0.2,
                    "combined_score": None,
                }
            ],
            "vector": [{"chunk_id": "chunk-bluup", "vector_score": 0.74, "keyword_score": None, "combined_score": None}],
            "keyword": [{"chunk_id": "chunk-lex", "vector_score": None, "keyword_score": 0.51, "combined_score": None}],
            "merged": [
                {
                    "chunk_id": "chunk-bluup",
                    "vector_score": 0.74,
                    "keyword_score": 0.2,
                    "combined_score": None,
                }
            ],
            "raw_count": 1,
            "raw_top_score": 0.7421,
            "raw_score_min": 0.7421,
            "raw_score_max": 0.7421,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dimension": 1024,
            "retrieval_query": "what is Bluup",
            "retrieval_k": 6,
        },
        "reranking": {
            "enabled": True,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "input_count": 1,
            "input": [
                {
                    "chunk_id": "chunk-bluup",
                    "vector_score": 0.74,
                    "keyword_score": 0.2,
                    "combined_score": None,
                    "section_path": "Products > Bluup",
                }
            ],
            "candidates": [
                {
                    "rank": 1,
                    "chunk_id": "chunk-bluup",
                    "score": 0.81,
                    "rerank_score": 0.81,
                    "original_retrieval_rank": 1,
                    "section_path": "Products > Bluup",
                    "text_preview": "Bluup is a hearing aid.",
                }
            ],
        },
        "final_context": {
            "context_count": 1,
            "chunks": [
                {
                    "chunk_id": "chunk-bluup",
                    "section_path": ["Products", "Bluup"],
                    "score": 0.81,
                    "text": "Bluup is a hearing aid.",
                    "title": "Bluup",
                }
            ],
        },
        "generation": {
            "provider": "litellm",
            "model": "openai/Qwen/Qwen2.5-7B-Instruct",
            "temperature": 0.0,
            "max_tokens": 512,
            "latency_ms": 12.0,
            "llm_call_count": 1,
            "system_prompt": "You are a helpful representative. Authorization: Bearer secret-token",
            "user_prompt": "User question:\nwhat is Bluup\nemail ada@example.com",
            "knowledge_context": "Bluup is a hearing aid.",
            "raw_output": '{"grounded": true, "answer": "Bluup is a hearing aid.", "source_ids": ["chunk-bluup"]}',
            "generation_result": {
                "grounded": True,
                "answer": "Bluup is a hearing aid.",
                "source_ids": ["chunk-bluup"],
            },
        },
        "grounding": {
            "grounded_returned": True,
            "validator_result": True,
            "validator_reason": "ok",
            "returned_source_ids": ["chunk-bluup"],
            "valid_source_ids": ["chunk-bluup"],
            "invalid_source_ids": [],
        },
        "response": {
            "final_mode": "KNOWLEDGE",
            "final_response": "Bluup is a hearing aid.",
            "source_ids": ["chunk-bluup"],
        },
        "state_after": {
            "conversation_goal": "NONE",
            "current_turn_intent": "KNOWLEDGE",
            "current_product": "Bluup",
            "lead_state": "NONE",
            "support_state": "NONE",
        },
        "latency": {
            "guardrail_ms": 1.2,
            "router_ms": 2.0,
            "rewrite_ms": 0.4,
            "retrieval_ms": 8.0,
            "rerank_ms": 3.0,
            "generation_ms": 12.0,
            "total_ms": 30.0,
        },
    }


def test_console_contains_required_sections() -> None:
    text = format_chat_debug_console(_sample_payload())
    for section in REQUIRED_SECTIONS:
        assert section in text, section


def test_console_prints_hybrid_scores_or_not_available() -> None:
    text = format_chat_debug_console(_sample_payload())
    assert "vector_score: 0.7400" in text
    assert "combined_score: NOT_AVAILABLE" in text
    assert "original_retrieval_rank: 1" in text


def test_console_redacts_pii_and_secrets() -> None:
    text = format_chat_debug_console(_sample_payload())
    assert "9876543210" not in text
    assert "ada@example.com" not in text
    assert "secret-token" not in text
    assert "[phone]" in text
    assert "[email]" in text
    assert "[redacted]" in text or "[token]" in text


def test_console_accepts_chat_trace_object() -> None:
    trace = ChatTrace()
    trace.request = {"conversation_id": "c1", "trace_id": "01TESTTRACE00000000000000", "user_message": "hi"}
    text = format_chat_debug_console(trace)
    assert "CHAT DEBUG" in text
    assert "c1" in text
    assert "CHAT TRACE" in format_chat_trace_text(trace.to_dict())


def test_first_failure_from_evidence_only() -> None:
    payload = _sample_payload()
    assert classify_first_failure(payload) == "NONE"
    payload["guardrail"]["blocked"] = True
    assert classify_first_failure(payload) == "GUARDRAIL"
    payload = _sample_payload()
    payload["retrieval"]["candidates"] = []
    payload["retrieval"]["vector"] = []
    payload["retrieval"]["keyword"] = []
    payload["retrieval"]["merged"] = []
    payload["retrieval"]["raw_count"] = 0
    payload["final_context"]["chunks"] = []
    payload["final_context"]["context_count"] = 0
    assert classify_first_failure(payload) == "RETRIEVAL"
    payload = _sample_payload()
    payload["grounding"]["validator_result"] = False
    payload["grounding"]["validator_reason"] = "empty_source_ids"
    assert classify_first_failure(payload) == "GROUNDING"
    payload = _sample_payload()
    payload["retrieval"]["candidates"][0]["chunk_id"] = "chunk-correct"
    payload["reranking"]["candidates"][0]["chunk_id"] = "chunk-other"
    payload["final_context"]["chunks"][0]["chunk_id"] = "chunk-other"
    assert classify_first_failure(payload) == "RERANKER"


def test_debug_console_disabled_does_not_print(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(settings, "chat_trace_enabled", False)
    monkeypatch.setattr(settings, "chat_debug_console", False)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("debug-off", "What is TINY?")
    captured = capsys.readouterr()
    assert "CHAT DEBUG" not in captured.out
    assert result.trace.get("trace_id")
    assert list(tmp_path.glob("*.json")) == []


def test_debug_console_enabled_prints_same_pipeline(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(settings, "chat_trace_enabled", False)
    monkeypatch.setattr(settings, "chat_debug_console", True)
    monkeypatch.setattr(settings, "chat_trace_include_full_context", True)
    monkeypatch.setattr(settings, "chat_trace_include_prompt", True)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("debug-on", "What is TINY?")
    captured = capsys.readouterr()
    assert "CHAT DEBUG" in captured.out
    assert "1. GUARDRAIL" in captured.out
    assert "DIAGNOSTIC SUMMARY" in captured.out
    assert result.trace.get("trace_id")
    assert (tmp_path / f"{result.trace['trace_id']}.json").exists()


def test_print_helper_writes_to_stream() -> None:
    stream = StringIO()
    print_chat_debug_console(_sample_payload(), stream=stream)
    assert "CHAT DEBUG" in stream.getvalue()


def test_raw_llm_output_is_visible() -> None:
    text = format_chat_debug_console(_sample_payload())
    assert '"grounded": true' in text
    assert "chunk-bluup" in text


def test_existing_chat_trace_shape_still_valid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "chat_trace_enabled", True)
    monkeypatch.setattr(settings, "chat_debug_console", False)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("compat-trace", "What is TINY?")
    from app.services.diagnostics.report import load_chat_trace

    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    for key in (
        "request",
        "guardrail",
        "turn_understanding",
        "retrieval",
        "reranking",
        "final_context",
        "generation",
        "grounding",
        "response",
        "latency",
    ):
        assert key in payload
    summary = diagnostic_summary(payload)
    assert summary["first_failure"] in {
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
    }
    assert "CHAT TRACE" in format_chat_trace_text(payload)
