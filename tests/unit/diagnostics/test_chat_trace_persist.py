from app.services.diagnostics.models import ChatTrace
from app.helpers.chat_trace_persist import chat_turn_fields, retrieval_layer_hit_fields


def _sample_trace() -> ChatTrace:
    trace = ChatTrace()
    trace.request = {
        "trace_id": "trace-1",
        "conversation_id": "conv-1",
        "user_message": "What is TINY?",
    }
    trace.turn_understanding = {"turn_intent": "KNOWLEDGE"}
    trace.query = {"rewritten_query": "what is tiny"}
    trace.response = {"final_response": "TINY is compact."}
    trace.tool_execution = {
        "tool_executed": True,
        "tool_name": "create_lead",
        "success": True,
        "lead_id": "lead-abc",
        "ticket_id": None,
        "latency_ms": 4.2,
    }
    trace.latency = {
        "guardrail_ms": 1.0,
        "routing_ms": 2.0,
        "rewrite_ms": 3.0,
        "retrieval_ms": 40.0,
        "generation_ms": 50.0,
        "tool_ms": 4.2,
        "total_ms": 100.0,
        "vector_ms": 10.5,
        "keyword_ms": 8.0,
        "merge_ms": 1.0,
        "rerank_ms": 12.0,
        "threshold_ms": 0.5,
    }
    trace.retrieval = {
        "backend": "llamaindex-hybrid",
        "vector": [
            {
                "rank": 1,
                "chunk_id": "c1",
                "document_id": "d1",
                "vector_score": 0.91,
                "title": "TINY",
                "section_path": "Products > TINY",
                "text_preview": "TINY is compact.",
            }
        ],
        "keyword": [
            {
                "rank": 1,
                "chunk_id": "c1",
                "keyword_score": 0.8,
                "title": "TINY",
                "text": "TINY is compact.",
            }
        ],
        "merged": [{"rank": 1, "chunk_id": "c1", "score": 0.91}],
    }
    trace.reranking = {
        "enabled": True,
        "name": "PassthroughReranker",
        "candidates": [
            {
                "rank": 1,
                "chunk_id": "c1",
                "rerank_score": 0.88,
                "original_retrieval_rank": 1,
            }
        ],
    }
    trace.final_context = {
        "chunks": [
            {
                "position": 1,
                "chunk_id": "c1",
                "document_id": "d1",
                "title": "TINY",
                "text": "TINY is compact.",
                "score": 0.88,
            }
        ]
    }
    return trace


def test_chat_turn_fields_copy_latency_and_ids() -> None:
    fields = chat_turn_fields(_sample_trace())
    assert fields["trace_id"] == "trace-1"
    assert fields["conversation_id"] == "conv-1"
    assert fields["lead_id"] == "lead-abc"
    assert fields["ticket_id"] is None
    assert fields["vector_ms"] == 10.5
    assert fields["keyword_ms"] == 8.0
    assert fields["merge_ms"] == 1.0
    assert fields["rerank_ms"] == 12.0
    assert fields["threshold_ms"] == 0.5
    assert fields["total_ms"] == 100.0
    assert fields["backend"] == "llamaindex-hybrid"
    assert fields["reranker_name"] == "PassthroughReranker"


class _RecordingTraces:
    def __init__(self) -> None:
        self.saved: list[ChatTrace] = []

    def save(self, trace: ChatTrace) -> None:
        self.saved.append(trace)


def test_orchestrator_always_persists_trace_without_files(tmp_path, monkeypatch) -> None:
    from app.config import settings
    from tests.unit.conversation.fakes import make_orchestrator

    monkeypatch.setattr(settings, "chat_trace_enabled", False)
    monkeypatch.setattr(settings, "chat_debug_console", False)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    recorder = _RecordingTraces()
    orchestrator, *_ = make_orchestrator(traces=recorder)
    result = orchestrator.handle("persist-off", "What is TINY?")
    assert result.trace.get("trace_id")
    assert recorder.saved
    assert recorder.saved[0].trace_id == result.trace["trace_id"]
    assert list(tmp_path.glob("*.json")) == []


class _BrokenTraces:
    def save(self, trace: ChatTrace) -> None:
        raise RuntimeError("postgres down")


def test_trace_persist_failure_does_not_fail_turn() -> None:
    from tests.unit.conversation.fakes import make_orchestrator

    orchestrator, *_ = make_orchestrator(traces=_BrokenTraces())
    result = orchestrator.handle("persist-fail", "What is TINY?")
    assert result.response
    assert result.trace.get("trace_id")


def test_background_trace_save_does_not_block_handle() -> None:
    import threading
    import time

    from app.repositories.chat_trace import BackgroundChatTraceRepository
    from tests.unit.conversation.fakes import make_orchestrator

    started = threading.Event()
    release = threading.Event()
    saved: list[ChatTrace] = []

    class _Slow:
        def save(self, trace: ChatTrace) -> None:
            started.set()
            release.wait(timeout=5)
            saved.append(trace)

    orchestrator, *_ = make_orchestrator(traces=BackgroundChatTraceRepository(_Slow()))
    result = orchestrator.handle("bg-persist", "What is TINY?")
    assert result.response
    assert result.trace.get("trace_id")
    assert saved == []
    assert started.wait(timeout=2)
    release.set()
    deadline = time.perf_counter() + 2
    while not saved and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert saved
    assert saved[0].trace_id == result.trace["trace_id"]

