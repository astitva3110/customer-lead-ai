from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.services.diagnostics.ids import new_trace_id
from app.services.diagnostics.models import ChatTrace
from app.services.diagnostics.redact import redact_mapping, redact_text
from app.services.diagnostics.report import format_chat_trace_text, load_chat_trace, write_chat_trace
from tests.unit.conversation.fakes import FakeKnowledge, RecordingLLM, make_orchestrator
from tests.validation.harness import make_graph_stack


def _enable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "chat_trace_enabled", True)
    monkeypatch.setattr(settings, "chat_debug_console", False)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    monkeypatch.setattr(settings, "chat_trace_include_prompt", False)
    monkeypatch.setattr(settings, "chat_trace_include_full_context", False)


def test_1_trace_object_creation() -> None:
    trace = ChatTrace()
    assert trace.request == {}
    assert trace.tool_execution["tool_executed"] is False
    assert "generation" in trace.to_dict()


def test_2_trace_id_propagation(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-id", "What is TINY?")
    trace_id = result.trace.get("trace_id")
    assert trace_id
    assert (tmp_path / f"{trace_id}.json").exists()


def test_3_request_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-req", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    assert payload["request"]["conversation_id"] == "diag-req"
    assert payload["request"]["user_message"] == "What is TINY?"
    assert payload["request"]["trace_id"] == result.trace["trace_id"]


def test_4_state_redaction() -> None:
    assert redact_text("call me +91 9876543210") == "call me [phone]"
    assert "[email]" in redact_text("write ada@example.com please")
    redacted = redact_mapping({"phone": "+919876543210", "api_key": "sk-secret", "goal": "SALES"})
    assert redacted["phone"] == "[redacted]"
    assert redacted["api_key"] == "[redacted]"
    assert redacted["goal"] == "SALES"
    chunk_id = "42b8c32cd41504b2ba6e3133bf55f3e24d175a9876543210a6f4bb40f4a6a"
    assert redact_text(chunk_id) == chunk_id


def test_5_turn_understanding_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-turn", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    turn = payload["turn_understanding"]
    assert turn["turn_intent"] == "KNOWLEDGE"
    assert turn["needs_rag"] is True
    assert turn["method"] == "deterministic"


def test_6_query_rewrite_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    orchestrator.handle("diag-q", "What is TINY?")
    result = orchestrator.handle("diag-q", "What is its battery life?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    query = payload["query"]
    assert query["rewrite_executed"] is True
    assert "TINY" in (query["rewritten_query"] or "")
    assert query["original_user_query"] == "What is its battery life?"


def test_7_and_8_retrieval_and_rerank_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    knowledge = FakeKnowledge()
    orchestrator, *_ = make_graph_stack(knowledge=knowledge)
    result = orchestrator.handle("diag-ret", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    assert payload["retrieval"]["candidate_count"] >= 1
    assert payload["retrieval"]["candidates"]
    assert "enabled" in payload["reranking"]
    assert knowledge.queries == ["What is TINY?"]


def test_9_final_context_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-ctx", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    context = payload["final_context"]
    assert context["context_count"] >= 1
    assert context["chunks"][0]["chunk_id"]
    assert context["chunks"][0]["text"]


def test_10_generation_metadata_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-gen", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    generation = payload["generation"]
    assert generation["provider"] == "litellm"
    assert generation["temperature"] == 0.0
    assert generation["llm_call_count"] == 1
    assert generation["latency_ms"] is not None


def test_11_grounding_result_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    llm = RecordingLLM(
        json.dumps({"grounded": True, "answer": "ok", "source_ids": ["does-not-exist"]})
    )
    orchestrator, *_ = make_orchestrator(llm=llm)
    result = orchestrator.handle("diag-ground", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    grounding = payload["grounding"]
    assert grounding["grounded_returned"] is True
    assert grounding["validator_result"] is False
    assert grounding["validator_reason"] == "source_id_not_found"
    assert grounding["fallback_triggered"] is True
    assert result.sources == []


def test_12_tool_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, _, _, lead, *_ = make_orchestrator()
    cid = "diag-tool"
    orchestrator.handle(cid, "I want to buy TINY.")
    orchestrator.handle(cid, "Please call me.")
    orchestrator.handle(cid, "+91 9876543210")
    orchestrator.handle(cid, "Ada")
    result = orchestrator.handle(cid, "Noida")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    assert payload["tool_execution"]["tool_executed"] is True
    assert payload["tool_execution"]["tool_name"] == "create_lead"
    assert payload["tool_execution"]["success"] is True
    assert lead.leads


def test_13_state_after_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-after", "I want to buy TINY.")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    after = payload["state_after"]
    assert after["conversation_goal"] == "LEAD"
    assert after["current_product"] == "TINY"


def test_14_latency_capture(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-lat", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    latency = payload["latency"]
    assert latency["total_ms"] is not None
    assert latency["generation_ms"] is not None
    assert "retrieval_ms" in latency


def test_15_error_capture(monkeypatch, tmp_path: Path) -> None:
    class _Boom:
        is_configured = True

        def complete(self, *args, **kwargs):
            raise RuntimeError("litellm down")

    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator(llm=_Boom())
    result = orchestrator.handle("diag-err", "What is TINY?")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    assert payload["errors"]
    assert payload["errors"][0]["component"] == "generation"
    assert result.response


def test_16_disabled_produces_no_files_or_extra_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "chat_trace_enabled", False)
    monkeypatch.setattr(settings, "chat_debug_console", False)
    monkeypatch.setattr(settings, "chat_trace_output_dir", tmp_path)
    knowledge = FakeKnowledge()
    llm = RecordingLLM()
    orchestrator, *_ = make_orchestrator(knowledge=knowledge, llm=llm)
    result = orchestrator.handle("diag-off", "What is TINY?")
    assert result.trace.get("trace_id")
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.txt")) == []
    assert knowledge.queries == ["What is TINY?"]
    assert llm.call_count == 1


def test_17_pii_redaction_in_files(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-pii", "My number is +91 9876543210 and email ada@example.com")
    raw = (tmp_path / f"{result.trace['trace_id']}.json").read_text(encoding="utf-8")
    text = (tmp_path / f"{result.trace['trace_id']}.txt").read_text(encoding="utf-8")
    assert "9876543210" not in raw
    assert "ada@example.com" not in raw
    assert "9876543210" not in text
    assert "sk-" not in raw


def test_18_and_19_json_and_txt_reports(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-fmt", "What is TINY?")
    trace_id = result.trace["trace_id"]
    json_path = tmp_path / f"{trace_id}.json"
    txt_path = tmp_path / f"{trace_id}.txt"
    assert json_path.exists()
    assert txt_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = txt_path.read_text(encoding="utf-8")
    assert payload["request"]["conversation_id"] == "diag-fmt"
    assert "CHAT TRACE" in text
    assert "TURN" in text
    assert "FINAL RESPONSE" in text


def test_20_inspect_chat_trace_cli(monkeypatch, tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    result = orchestrator.handle("diag-cli", "What is TINY?")
    repo = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_chat_trace.py",
            "--trace-id",
            result.trace["trace_id"],
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "CHAT TRACE" in proc.stdout
    proc_json = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_chat_trace.py",
            "--trace-id",
            result.trace["trace_id"],
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
    )
    assert proc_json.returncode == 0, proc_json.stderr
    payload = json.loads(proc_json.stdout)
    assert payload["request"]["conversation_id"] == "diag-cli"


def test_explain_grounding_does_not_change_validator() -> None:
    allowed = {"chunk_1"}
    payload = {"grounded": True, "answer": "ok", "source_ids": ["missing"]}
    from app.helpers.generation_validate import explain_generation_payload, validate_generation_payload

    assert validate_generation_payload(payload, allowed) is None
    explained = explain_generation_payload(payload, allowed)
    assert explained["validator_reason"] == "source_id_not_found"
    assert explained["invalid_source_ids"] == ["missing"]
    assert explained["valid_source_ids"] == []


def test_new_trace_id_is_unique() -> None:
    assert new_trace_id() != new_trace_id()
    assert len(new_trace_id()) == 26


def test_write_chat_trace_round_trip(tmp_path: Path) -> None:
    trace = ChatTrace()
    trace.request = {"conversation_id": "c1", "trace_id": "01TESTTRACE00000000000000", "user_message": "hi"}
    json_path, txt_path = write_chat_trace(trace, tmp_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["request"]["conversation_id"] == "c1"
    assert "CHAT TRACE" in txt_path.read_text(encoding="utf-8")
    assert "CHAT TRACE" in format_chat_trace_text(loaded)


def test_noisy_query_is_prepared_for_retrieval(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    knowledge = FakeKnowledge()
    orchestrator, *_ = make_graph_stack(knowledge=knowledge)
    result = orchestrator.handle("diag-noisy", "ok what is tiny")
    payload = load_chat_trace(result.trace["trace_id"], tmp_path)
    assert payload["query"]["rewrite_executed"] is True
    assert payload["query"]["rewritten_query"] == "what is tiny"
    assert knowledge.queries[-1] == "what is tiny"


def test_enabled_does_not_add_retrieval_or_llm_calls(monkeypatch, tmp_path: Path) -> None:
    _enable(monkeypatch, tmp_path)
    knowledge = FakeKnowledge()
    llm = RecordingLLM()
    orchestrator, *_ = make_orchestrator(knowledge=knowledge, llm=llm)
    orchestrator.handle("diag-once", "What is TINY?")
    assert knowledge.queries == ["What is TINY?"]
    assert llm.call_count == 1


def test_debug_trace_id_only_when_enabled(monkeypatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.dependencies import get_orchestrator

    _enable(monkeypatch, tmp_path)
    orchestrator, *_ = make_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        client = TestClient(app)
        response = client.post("/chat", json={"message": "What is TINY?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["debug_trace_id"]
        assert (tmp_path / f"{payload['debug_trace_id']}.json").exists()
    finally:
        app.dependency_overrides.clear()
