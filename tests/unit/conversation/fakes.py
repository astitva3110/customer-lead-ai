from __future__ import annotations

import json

from app.services.generation.generation_service import GenerationService
from app.services.generation.models import INSUFFICIENT_INFORMATION_MESSAGE
from app.repositories.memory_lead import InMemoryLeadAdapter
from app.repositories.memory_ticket import InMemoryTicketAdapter
from app.repositories.conversation import InMemoryConversationRepository
from app.graph.factory import build_conversation_orchestrator


class RecordingLLM:
    def __init__(self, output: str | None = None, *, configured: bool = True) -> None:
        self.output = output
        self.call_count = 0
        self.is_configured = configured
        self.last_temperature: float | None = None
        self.last_user: str = ""

    def complete(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int | None = None) -> str:
        del system, max_tokens
        self.call_count += 1
        self.last_temperature = temperature
        self.last_user = user
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
