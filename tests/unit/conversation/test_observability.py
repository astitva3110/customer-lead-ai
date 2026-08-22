import logging

from tests.unit.conversation.fakes import make_orchestrator


def test_trace_logs_mode_and_latency_without_phone(caplog) -> None:
    orchestrator, *_ = make_orchestrator()
    with caplog.at_level(logging.INFO, logger="conversation.trace"):
        orchestrator.handle("obs-1", "I want to buy Radius M16.")
        orchestrator.handle("obs-1", "+91 9876543210")
    text = caplog.text
    assert "obs-1" in text
    assert "LEAD" in text
    assert "total_latency_ms" in text
    assert "9876543210" not in text
    assert "+91" not in text
