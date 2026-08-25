import sys
from types import SimpleNamespace

from app.config import Settings
from app.providers.llm.litellm_provider import LiteLLMProvider


def test_litellm_adapter_single_completion(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content='{"grounded": true, "answer": "ok", "source_ids": ["c1"]}')
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))
    settings = Settings(
        generation_model="openai/Qwen/Qwen3-1.7B",
        generation_api_base="http://localhost:8001/v1",
        generation_json_mode=True,
        generation_api_key="dummy-key",
        generation_timeout_seconds=45.0,
        generation_num_retries=0,
        generation_extra_body='{"think":false}',
    )
    provider = LiteLLMProvider(settings)
    text = provider.complete("sys", "user", temperature=0.0, max_tokens=512)
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/Qwen/Qwen3-1.7B"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["timeout"] == 45.0
    assert calls[0]["num_retries"] == 0
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["api_base"] == "http://localhost:8001/v1"
    assert calls[0]["extra_body"] == {"think": False}
    assert '"grounded": true' in text


def test_openai_api_settings_use_same_adapter(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))
    settings = Settings(
        generation_model="openai/gpt-4o-mini",
        generation_api_base="",
        generation_api_key="sk-test",
        generation_temperature=0.2,
        generation_max_tokens=256,
        generation_timeout_seconds=30.0,
        generation_json_mode=True,
    )
    text = LiteLLMProvider(settings).complete("sys", "user")
    assert text == "ok"
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    assert "api_base" not in calls[0]
    assert calls[0]["api_key"] == "sk-test"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["max_tokens"] == 256
    assert calls[0]["timeout"] == 30.0


def test_litellm_adapter_records_token_usage(monkeypatch) -> None:
    def fake_completion(**kwargs):
        del kwargs
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
        return SimpleNamespace(choices=[choice], usage=usage)

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=fake_completion))
    from app.services.diagnostics.recorder import TraceSession

    settings = Settings(generation_model="openai/gpt-4o-mini", generation_api_key="sk-test")
    state = type(
        "State",
        (),
        {
            "conversation_id": "c1",
            "conversation_goal": "",
            "current_turn_intent": "",
            "lead_stage": "",
            "user_context": {},
            "lead_workflow": "",
            "support_workflow": "",
            "awaiting_field": "",
            "conversation_history": [],
            "product": "",
        },
    )()
    session = TraceSession.start(state, "hello")
    try:
        LiteLLMProvider(settings).complete("sys", "user")
    finally:
        session.close()
    assert session.trace.observability["token_usage"]["input_tokens"] == 11
    assert session.trace.observability["token_usage"]["output_tokens"] == 7
