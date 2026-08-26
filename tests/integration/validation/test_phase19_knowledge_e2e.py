from __future__ import annotations

from app.config import settings
from app.services.conversation.models import ChatMode
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.providers.knowledge.knowledge_service import LlamaIndexKnowledgeService
from tests.validation.harness import make_graph_stack

KNOWLEDGE_QUERIES = [
    "What is BTE?",
    "What is TINY?",
    "What is Bluup?",
    "What is the address of Earkart?",
    "What is the return policy?",
    "When should I expect delivery?",
    "What is the warranty policy?",
    "How do hearing aids work?",
    "What are the different types of hearing aids?",
    "What are Earkart's products?",
    "Tell me about Earkart's hearing aids.",
    "what is this bte?",
    "when to except the deeeliver of item",
]


def _orchestrator(production_hybrid):
    knowledge = LlamaIndexKnowledgeService(
        production_hybrid["hybrid"],
        retrieval_version=production_hybrid["config"].embedding_version,
    )
    return make_graph_stack(knowledge=knowledge)


def test_knowledge_queries_use_hybrid_path(production_hybrid) -> None:
    orchestrator, recorder, *_ = _orchestrator(production_hybrid)
    failures = []
    for index, query in enumerate(KNOWLEDGE_QUERIES):
        result = orchestrator.handle(f"know-{index}", query)
        meta = result.retrieval_metadata or {}
        if result.mode != ChatMode.KNOWLEDGE:
            failures.append((query, f"mode={result.mode}"))
            continue
        if not meta.get("vector_candidate_count") and not meta.get("keyword_candidate_count"):
            failures.append((query, "no hybrid candidate counts"))
            continue
        if result.response == INSUFFICIENT_INFORMATION_MESSAGE:
            if result.sources:
                failures.append((query, "fallback with sources"))
            continue
        if not result.retrieved_chunk_ids:
            failures.append((query, "empty context for answered query"))
        if not result.sources:
            failures.append((query, "answered without sources"))
        source_ids = {item.get("chunk_id") for item in result.sources}
        if source_ids - set(result.retrieved_chunk_ids):
            failures.append((query, "source not in retrieved context"))
    assert not failures, failures
    assert recorder.call_count >= 1


def test_why_buy_spec_expects_knowledge(production_hybrid) -> None:
    orchestrator, recorder, *_ = _orchestrator(production_hybrid)
    result = orchestrator.handle("why-buy", "Why should I buy from Earkart?")
    assert result.mode == ChatMode.KNOWLEDGE, (
        f"SPEC expects KNOWLEDGE; implementation routed to {result.mode}"
    )
    assert recorder.call_count >= 1


def test_contextual_tiny_battery(production_hybrid) -> None:
    orchestrator, recorder, *_ = _orchestrator(production_hybrid)
    first = orchestrator.handle("ctx-tiny", "What is TINY?")
    second = orchestrator.handle("ctx-tiny", "What is its battery life?")
    assert first.mode == ChatMode.KNOWLEDGE
    assert second.mode == ChatMode.KNOWLEDGE
    assert second.query_rewritten == "What is the battery life of TINY?"
    assert recorder.generation_call_count == 2


def test_corpus_gaps_do_not_fabricate(production_hybrid) -> None:
    orchestrator, _, *_ = _orchestrator(production_hybrid)
    for index, query in enumerate(("What is the battery life in hours?", "What is the Signia hearing aid?")):
        result = orchestrator.handle(f"gap-{index}", query)
        assert result.mode == ChatMode.KNOWLEDGE
        assert result.response == INSUFFICIENT_INFORMATION_MESSAGE
        assert result.sources == []
