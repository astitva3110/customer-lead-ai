from __future__ import annotations

import logging

import pytest

from app import dependencies as deps


@pytest.fixture(autouse=True)
def _clear_orchestrator_caches() -> None:
    deps.get_orchestrator.cache_clear()
    deps.get_chat_trace_repository.cache_clear()
    deps.get_background_chat_trace_repository.cache_clear()
    deps.get_chat_history_service.cache_clear()
    yield
    deps.get_orchestrator.cache_clear()
    deps.get_chat_trace_repository.cache_clear()
    deps.get_background_chat_trace_repository.cache_clear()
    deps.get_chat_history_service.cache_clear()


def test_get_orchestrator_logs_and_reraises_retriever_failure(monkeypatch, caplog) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("postgres down")

    monkeypatch.setattr("app.kb.retrieval.service.RetrievalService.from_config", boom)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="postgres down"):
            deps.get_orchestrator()
    assert "knowledge retriever initialization failed" in caplog.text


def test_get_orchestrator_does_not_swallow_config_errors(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise FileNotFoundError("missing yaml")

    monkeypatch.setattr(
        "app.kb.evaluation.retrieval_config.RetrievalConfig.from_yaml",
        boom,
    )
    with pytest.raises(FileNotFoundError, match="missing yaml"):
        deps.get_orchestrator()
