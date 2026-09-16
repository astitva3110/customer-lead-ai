from __future__ import annotations

from types import SimpleNamespace

from app.helpers.token_usage import attach_token_usage_summary, summarize_chat_trace_tokens
from app.services.diagnostics.models import ChatTrace


def test_summarize_chat_trace_tokens_totals_and_breakdown() -> None:
    chat_trace = ChatTrace(
        observability={
            "token_usage": {
                "input_tokens": 1200,
                "output_tokens": 180,
                "total_tokens": 1380,
            }
        },
        semantic_router={
            "input_tokens": 400,
            "output_tokens": 60,
            "total_tokens": 460,
        },
        generation={
            "input_token_count": 800,
            "output_token_count": 120,
        },
    )
    summary = summarize_chat_trace_tokens(chat_trace)
    assert summary["input_tokens"] == 1200
    assert summary["output_tokens"] == 180
    assert summary["total_tokens"] == 1380
    assert summary["breakdown"]["semantic_router"]["total_tokens"] == 460
    assert summary["breakdown"]["generation"]["total_tokens"] == 920


def test_attach_token_usage_summary_copies_to_state_trace() -> None:
    chat_trace = ChatTrace(
        observability={"token_usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}},
        semantic_router={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
    )
    state_trace: dict = {}
    attach_token_usage_summary(state_trace, chat_trace)
    assert state_trace["token_usage"]["total_tokens"] == 60
    assert state_trace["token_usage_breakdown"]["semantic_router"]["total_tokens"] == 60


def test_summarize_chat_trace_tokens_from_dict_payload() -> None:
    payload = {
        "observability": {"token_usage": {"input_tokens": 10, "output_tokens": 5}},
        "semantic_router": {},
        "generation": {"input_token_count": 10, "output_token_count": 5},
    }
    summary = summarize_chat_trace_tokens(payload)
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 5
    assert summary["total_tokens"] == 15
    assert summary["breakdown"]["generation"]["total_tokens"] == 15


def test_summarize_chat_trace_tokens_handles_missing_usage() -> None:
    summary = summarize_chat_trace_tokens(SimpleNamespace())
    assert summary["input_tokens"] is None
    assert summary["output_tokens"] is None
    assert summary["total_tokens"] is None
