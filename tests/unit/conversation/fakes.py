from __future__ import annotations

import json

from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.repositories.memory_lead import InMemoryLeadAdapter
from app.repositories.memory_ticket import InMemoryTicketAdapter
from app.repositories.conversation import InMemoryConversationRepository
from app.graph.factory import build_conversation_orchestrator
from app.helpers.semantic_router import SEMANTIC_ROUTER_MARKER


class RecordingLLM:
    def __init__(self, output: str | None = None, *, configured: bool = True, router_output: str | None = None) -> None:
        self.output = output
        self.router_output = router_output
        self.call_count = 0
        self.router_call_count = 0
        self.is_configured = configured
        self.last_temperature: float | None = None
        self.last_user: str = ""
        self.last_system: str = ""

    @property
    def generation_call_count(self) -> int:
        return self.call_count - self.router_call_count

    def complete(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int | None = None) -> str:
        del max_tokens
        self.call_count += 1
        self.last_temperature = temperature
        self.last_user = user
        self.last_system = system
        if SEMANTIC_ROUTER_MARKER in (system or ""):
            self.router_call_count += 1
            if self.router_output is not None:
                return self.router_output
            return json.dumps(
                {
                    "canonical_query": "",
                    "route": "KNOWLEDGE",
                    "product": None,
                    "sales_interest": False,
                    "diverge": False,
                    "sub_questions": [],
                    "confidence": 0.0,
                }
            )
        if self.output is not None:
            return self.output
        if "current_message" in user:
            try:
                payload = json.loads(user)
                message = str(payload.get("current_message") or "")
                product = str(payload.get("product") or "that")
            except json.JSONDecodeError:
                message = user
                product = "that"
            return json.dumps({"answer": f"I can help you with {product}. ({message[:48]})"})
        return json.dumps(
            {
                "grounded": True,
                "answer": "Radius M16 is a hearing aid.",
                "source_ids": ["chunk-1"],
            }
        )


class FakeKnowledge:
    def __init__(self, chunks: list[dict] | None = None) -> None:
        self.queries: list[str] = []
        self.chunks = chunks

    def retrieve_knowledge(self, query: str, *, context: dict | None = None) -> dict:
        del context
        self.queries.append(query)
        chunks = self.chunks
        if chunks is None:
            chunks = [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "section_path": ["Products", "Radius M16"],
                    "page_number": 1,
                    "title": "Radius M16",
                    "content_type": "paragraph",
                    "text": "Radius M16 is a rechargeable hearing aid.",
                    "score": 0.91,
                }
            ]
        return {
            "chunks": chunks,
            "scores": [float(item.get("score") or 0.0) for item in chunks],
            "metadata": [{} for _ in chunks],
            "query": query,
            "retrieval_version": "v2",
            "vector_candidate_count": 20,
            "keyword_candidate_count": 20,
            "merged_candidate_count": 30,
            "reranked_count": 30,
        }


def make_orchestrator(
    *,
    knowledge: FakeKnowledge | None = None,
    llm: RecordingLLM | None = None,
    lead_tool: InMemoryLeadAdapter | None = None,
    ticket_tool: InMemoryTicketAdapter | None = None,
    traces=None,
):
    knowledge = knowledge or FakeKnowledge()
    llm = llm or RecordingLLM()
    lead_tool = lead_tool or InMemoryLeadAdapter()
    ticket_tool = ticket_tool or InMemoryTicketAdapter()
    store = InMemoryConversationRepository()
    orchestrator = build_conversation_orchestrator(
        knowledge=knowledge,
        generation=GenerationService(
            llm,
            temperature=0.0,
            conversation_temperature=0.4,
            max_tokens=512,
        ),
        lead_tool=lead_tool,
        ticket_tool=ticket_tool,
        store=store,
        model_used="test-qwen",
        traces=traces,
    )
    return orchestrator, knowledge, llm, lead_tool, ticket_tool, store
