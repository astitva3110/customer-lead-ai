from __future__ import annotations

import json

from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.kb.evaluation.models import RankedHit


class _RecordingLLM:
    def __init__(self, output: str | Exception, *, configured: bool = True) -> None:
        self.output = output
        self.call_count = 0
        self.is_configured = configured
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None

    def complete(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int | None = None) -> str:
        self.call_count += 1
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _hit(chunk_id: str, text: str, document_id: str = "doc-1") -> RankedHit:
    return RankedHit(
        rank=1,
        similarity=0.91,
        chunk_id=chunk_id,
        document_id=document_id,
        section_path=["About"],
        token_count=12,
        content_type="paragraph",
        text=text,
        page_number=1,
        document_title="Pricelist",
    )


def _service(output: str | Exception, **kwargs) -> tuple[GenerationService, _RecordingLLM]:
    llm = _RecordingLLM(output, **kwargs)
    return GenerationService(llm, temperature=0.0, max_tokens=512), llm


def test_grounded_answer_accepts_known_chunk() -> None:
    service, llm = _service(
        json.dumps({"grounded": True, "answer": "Earkart provides hearing care.", "source_ids": ["chunk_1"]})
    )
    result = service.generate("What is Earkart?", [_hit("chunk_1", "Earkart is a digital-first platform.")])
    assert llm.call_count == 1
    assert llm.last_temperature == 0.0
    assert result.grounded is True
    assert result.answer == "Earkart provides hearing care."
    assert result.source_ids == ["chunk_1"]


def test_unsupported_answer_is_safe_fallback() -> None:
    service, llm = _service(
        json.dumps(
            {
                "grounded": False,
                "answer": INSUFFICIENT_INFORMATION_MESSAGE,
                "source_ids": [],
            }
        )
    )
    result = service.generate("Where are New York stores?", [_hit("chunk_1", "HQ is in Noida.")])
    assert llm.call_count == 1
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.source_ids == []


def test_exact_numeric_mismatch_mock_is_ungrounded() -> None:
    service, llm = _service(
        json.dumps({"grounded": False, "answer": INSUFFICIENT_INFORMATION_MESSAGE, "source_ids": []})
    )
    result = service.generate(
        "What is FAME at ₹8,900?",
        [_hit("chunk_1", "FAME MRP is ₹10,999.")],
    )
    assert llm.call_count == 1
    assert result.grounded is False
    assert "10,999" not in result.answer
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE


def test_unknown_source_id_fails_closed_no_answer() -> None:
    service, llm = _service(
        json.dumps({"grounded": True, "answer": "The MRP is ₹8,900.", "source_ids": ["does-not-exist"]})
    )
    result = service.generate("What is FAME at ₹8,900?", [_hit("chunk_1", "FAME MRP is ₹8,900.")])
    assert llm.call_count == 1
    assert result.grounded is False
    assert result.source_ids == []
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert "8,900" not in result.answer or result.answer == INSUFFICIENT_INFORMATION_MESSAGE


def test_omitted_source_ids_fails_closed_one_call() -> None:
    service, llm = _service(json.dumps({"grounded": True, "answer": "Bluup is an earplug."}))
    result = service.generate("what is Bluup", [_hit("chunk_bluup", "Bluup is a noise-reducing earplug.")])
    assert llm.call_count == 1
    assert result.grounded is False
    assert result.source_ids == []
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE


def test_invalid_source_id_fails_closed_one_call() -> None:
    service, llm = _service(
        json.dumps({"grounded": True, "answer": "TINY is compact.", "source_ids": ["not-in-context"]})
    )
    result = service.generate("what is TINY", [_hit("chunk_tiny", "TINY is a compact hearing aid.")])
    assert llm.call_count == 1
    assert result.grounded is False
    assert result.source_ids == []
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    service, llm = _service("FAME costs ₹10,999.")
    result = service.generate("What is FAME at ₹8,900?", [_hit("chunk_1", "FAME MRP is ₹10,999.")])
    assert llm.call_count == 1
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert "10,999" not in result.answer


def test_markdown_json_is_accepted() -> None:
    raw = """```json
{"grounded": true, "answer": "Earkart is a hearing-care platform.", "source_ids": ["chunk_1"]}
```"""
    service, llm = _service(raw)
    result = service.generate("What is Earkart?", [_hit("chunk_1", "About Earkart...")])
    assert llm.call_count == 1
    assert result.grounded is True
    assert result.source_ids == ["chunk_1"]


def test_empty_hits_does_not_call_llm() -> None:
    service, llm = _service("should not be used")
    result = service.generate("anything", [])
    assert llm.call_count == 0
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE


def test_llm_exception_fails_closed_without_retry() -> None:
    service, llm = _service(RuntimeError("litellm down"))
    result = service.generate("What is Earkart?", [_hit("chunk_1", "About Earkart")])
    assert llm.call_count == 1
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE


def test_unconfigured_llm_does_not_call() -> None:
    service, llm = _service("unused", configured=False)
    result = service.generate("What is Earkart?", [_hit("chunk_1", "About Earkart")])
    assert llm.call_count == 0
    assert result.grounded is False


def test_trace_records_omitted_source_ids(monkeypatch) -> None:
    from app.config import settings
    from app.services.conversation.models import ConversationState
    from app.services.diagnostics.recorder import TraceSession, current_trace

    monkeypatch.setattr(settings, "chat_trace_enabled", True)
    state = ConversationState(conversation_id="trace-omit", user_message="what is Bluup")
    session = TraceSession.start(state, "what is Bluup")
    try:
        service, llm = _service(json.dumps({"grounded": True, "answer": "Bluup is an earplug."}))
        result = service.generate("what is Bluup", [_hit("chunk_bluup", "Bluup is a noise-reducing earplug.")])
        trace = current_trace()
        assert llm.call_count == 1
        assert result.grounded is False
        assert trace is not None
        assert trace.generation["raw_model_output"]
        assert trace.generation["parsed_grounded"] is True
        assert trace.generation["parsed_answer"] == "Bluup is an earplug."
        assert trace.generation["parsed_source_ids"] is None
        assert trace.generation["validator_reason"] == "missing_fields"
        assert trace.grounding["validator_reason"] == "missing_fields"
    finally:
        session.close()


def test_trace_records_invalid_source_ids(monkeypatch) -> None:
    from app.config import settings
    from app.services.conversation.models import ConversationState
    from app.services.diagnostics.recorder import TraceSession, current_trace

    monkeypatch.setattr(settings, "chat_trace_enabled", True)
    state = ConversationState(conversation_id="trace-bad-id", user_message="what is TINY")
    session = TraceSession.start(state, "what is TINY")
    try:
        service, _ = _service(
            json.dumps({"grounded": True, "answer": "TINY is compact.", "source_ids": ["not-in-context"]})
        )
        result = service.generate("what is TINY", [_hit("chunk_tiny", "TINY is a compact hearing aid.")])
        trace = current_trace()
        assert result.grounded is False
        assert trace is not None
        assert trace.generation["parsed_source_ids"] == ["not-in-context"]
        assert trace.generation["invalid_source_ids"] == ["not-in-context"]
        assert trace.generation["validator_reason"] == "source_id_not_found"
        assert trace.grounding["validator_reason"] == "source_id_not_found"
    finally:
        session.close()
    from app.services.conversation.models import ConversationState

    llm = _RecordingLLM(json.dumps({"answer": "Absolutely. I can help you with TINY."}))
    service = GenerationService(llm, temperature=0.0, conversation_temperature=0.4, max_tokens=512)
    state = ConversationState(user_message="I want to buy TINY.", product="TINY")
    result = service.converse(state)
    assert llm.call_count == 1
    assert llm.last_temperature == 0.4
    assert "TINY" in result.response
    assert "please provide" not in result.response.lower()


def test_converse_does_not_repeat_last_assistant() -> None:
    from app.services.conversation.models import ConversationState

    repeated = "Absolutely. I can help you with that. Would you like to know anything about TINY before I arrange a callback?"
    llm = _RecordingLLM(json.dumps({"answer": repeated}))
    service = GenerationService(llm, conversation_temperature=0.4)
    state = ConversationState(
        user_message="yes",
        product="TINY",
        conversation_history=[{"role": "assistant", "content": repeated}],
    )
    result = service.converse(state)
    assert result.response != repeated
    assert "would you like to know anything about tiny before" not in result.response.lower()

