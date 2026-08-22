"""Call the existing conversation orchestrator. Does not replace the /chat pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import settings
from app.services.diagnostics.report import load_chat_trace


HandleFn = Callable[[str, str], "TurnOutcome"]


@dataclass
class TurnOutcome:
    conversation_id: str
    response: str
    trace_id: str
    trace: dict[str, Any]
    mode: str = ""
    error: str | None = None


_shared_lock = threading.Lock()
_shared_knowledge = None
_shared_generation = None


def enable_tracing(output_dir: Path, *, top_k: int = 10) -> None:
    settings.chat_trace_enabled = True
    settings.chat_debug_console = False
    settings.chat_trace_output_dir = output_dir
    settings.chat_trace_retrieval_top_k = top_k
    output_dir.mkdir(parents=True, exist_ok=True)


def shared_production_stack():
    """Reuse production LlamaIndex/LiteLLM wiring. New store per worker."""
    global _shared_knowledge, _shared_generation
    from app.services.generation.generation_service import GenerationService
    from app.providers.knowledge.knowledge_service import LlamaIndexKnowledgeService
    from app.providers.llm.factory import get_llm_provider
    from app.providers.retrieval.factory import build_knowledge_hybrid_retriever
    from app.kb.evaluation.retrieval_config import RetrievalConfig
    from app.kb.retrieval.service import RetrievalService

    with _shared_lock:
        if _shared_knowledge is None or _shared_generation is None:
            retrieval_yaml = Path(__file__).resolve().parents[2] / "configs" / "retrieval" / "v2.yaml"
            config = RetrievalConfig.from_yaml(retrieval_yaml)
            vector_service = RetrievalService.from_config(config)
            hybrid = build_knowledge_hybrid_retriever(config, vector_service=vector_service)
            _shared_knowledge = LlamaIndexKnowledgeService(
                hybrid,
                retrieval_version=config.embedding_version,
            )
            _shared_generation = GenerationService(
                get_llm_provider(settings),
                temperature=settings.generation_temperature,
                conversation_temperature=settings.generation_conversation_temperature,
                max_tokens=settings.generation_max_tokens,
            )
        return _shared_knowledge, _shared_generation


def build_eval_orchestrator():
    from app.services.conversation.orchestrator import ConversationOrchestrator
    from app.repositories.memory_lead import InMemoryLeadAdapter
    from app.repositories.memory_ticket import InMemoryTicketAdapter
    from app.repositories.conversation import InMemoryConversationRepository
    from app.graph.factory import build_conversation_orchestrator

    knowledge, generation = shared_production_stack()
    return build_conversation_orchestrator(
        knowledge=knowledge,
        generation=generation,
        lead_tool=InMemoryLeadAdapter(),
        ticket_tool=InMemoryTicketAdapter(),
        store=InMemoryConversationRepository(),
        model_used=settings.generation_model,
    )


class OrchestratorChatClient:
    """Existing orchestrator.handle path used by POST /chat."""

    def __init__(self, orchestrator=None) -> None:
        self._lock = threading.Lock()
        self._by_conversation: dict[str, Any] = {}
        self._orchestrator = orchestrator

    def _worker(self, conversation_id: str):
        if self._orchestrator is not None:
            return self._orchestrator
        with self._lock:
            current = self._by_conversation.get(conversation_id)
            if current is None:
                current = build_eval_orchestrator()
                self._by_conversation[conversation_id] = current
            return current

    def drop(self, conversation_id: str) -> None:
        with self._lock:
            self._by_conversation.pop(conversation_id, None)

    def handle(self, conversation_id: str, message: str) -> TurnOutcome:
        result = self._worker(conversation_id).handle(conversation_id, message)
        trace_id = str((result.trace or {}).get("trace_id") or "")
        trace: dict[str, Any] = {}
        if trace_id:
            try:
                trace = load_chat_trace(trace_id, settings.chat_trace_output_dir)
            except Exception:
                trace = {"trace_id": trace_id, "errors": [{"component": "trace_load", "error_message": "missing"}]}
        return TurnOutcome(
            conversation_id=result.conversation_id,
            response=result.response or "",
            trace_id=trace_id,
            trace=trace,
            mode=str(result.mode or ""),
        )
