from __future__ import annotations

from tests.unit.conversation.fakes import FakeKnowledge
from tests.validation.harness import latency_summary, make_graph_stack

KNOWLEDGE = [
    "What is TINY?",
    "What is BTE?",
    "What is Bluup?",
    "What is the warranty policy?",
    "What is the return policy?",
]
CONTEXTUAL = [
    ("What is TINY?", "What is its battery life?"),
    ("What is TINY?", "How does it work?"),
    ("What is Radius M16?", "What is its battery life?"),
    ("What is Bluup?", "What is its battery life?"),
    ("What is TINY?", "What is this?"),
]
LEAD = [
    "I want to buy Radius M16.",
    "+91 9876543210",
    "Ada",
    "Noida",
    "I want to buy TINY.",
]
SUPPORT = [
    "My hearing aid isn't working.",
    "Ada",
    "Radius M16",
    "+91 9876543210",
    "My Radius M16 stopped working.",
]


def _capture(handle, cid: str, message: str) -> dict:
    result = handle(cid, message)
    trace = dict(result.trace or {})
    meta = dict(result.retrieval_metadata or {})
    return {
        "mode": result.mode,
        "total_ms": trace.get("total_latency_ms"),
        "query_rewrite_ms": trace.get("query_rewrite_ms"),
        "retrieval_ms": trace.get("retrieval_ms"),
        "generation_ms": trace.get("generation_ms"),
        "query_rewrite_called": bool(result.query_rewritten),
        "llm_call_count": 1 if trace.get("generation_ms") is not None else 0,
        "retrieval_stage_ms": trace.get("retrieval_stage_ms"),
        "vector_count": meta.get("vector_candidate_count"),
        "keyword_count": meta.get("keyword_candidate_count"),
    }


def test_latency_and_llm_call_efficiency() -> None:
    knowledge = FakeKnowledge()
    orchestrator, recorder, *_ = make_graph_stack(knowledge=knowledge)
    simple = [_capture(orchestrator.handle, f"s{i}", query) for i, query in enumerate(KNOWLEDGE)]
    assert all(row["mode"] == "KNOWLEDGE" for row in simple)
    assert all(row["query_rewrite_called"] is False for row in simple)

    rewrite_counts = []
    contextual = []
    for index, (first, second) in enumerate(CONTEXTUAL):
        orchestrator.handle(f"c{index}", first)
        row = _capture(orchestrator.handle, f"c{index}", second)
        contextual.append(row)
        rewrite_counts.append(row["query_rewrite_called"])
    assert any(rewrite_counts)

    lead = []
    cid = "lat-lead"
    for message in LEAD[:4]:
        lead.append(_capture(orchestrator.handle, cid, message))
    lead.append(_capture(orchestrator.handle, "lat-lead-2", LEAD[4]))
    assert all(row["mode"] == "LEAD" for row in lead)

    support = []
    cid = "lat-sup"
    for message in SUPPORT[:4]:
        support.append(_capture(orchestrator.handle, cid, message))
    support.append(_capture(orchestrator.handle, "lat-sup-2", SUPPORT[4]))
    assert all(row["mode"] == "SUPPORT" for row in support)

    rag_calls = len(KNOWLEDGE) + len(CONTEXTUAL) * 2
    assert recorder.call_count >= rag_calls
    assert knowledge.queries[:5] == KNOWLEDGE
    summary = {
        "simple_knowledge": latency_summary([row["total_ms"] for row in simple if row["total_ms"] is not None]),
        "contextual_knowledge": latency_summary([row["total_ms"] for row in contextual if row["total_ms"] is not None]),
        "lead": latency_summary([row["total_ms"] for row in lead if row["total_ms"] is not None]),
        "support": latency_summary([row["total_ms"] for row in support if row["total_ms"] is not None]),
    }
    assert summary["simple_knowledge"]["n"] == 5
    assert summary["lead"]["n"] == 5
