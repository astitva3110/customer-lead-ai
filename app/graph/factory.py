from __future__ import annotations

from app.config import settings
from app.services.conversation.knowledge_flow import KnowledgeFlow
from app.services.conversation.lead_service import LeadService
from app.services.conversation.orchestrator import ConversationOrchestrator
from app.services.conversation.router import ChatRouter
from app.services.conversation.support_service import SupportService
from app.services.generation.generation_service import GenerationService
from app.interfaces.providers.business import LeadTool, TicketTool
from app.interfaces.providers.knowledge import KnowledgeService
from app.interfaces.repositories.chat_trace_repository import ChatTraceRepository
from app.interfaces.repositories.conversation_repository import ConversationRepository
from app.graph.chat_graph import build_chat_graph


def build_conversation_orchestrator(
    *,
    knowledge: KnowledgeService,
    generation: GenerationService,
    lead_tool: LeadTool,
    ticket_tool: TicketTool,
    store: ConversationRepository,
    model_used: str = "",
    traces: ChatTraceRepository | None = None,
) -> ConversationOrchestrator:
    graph = build_chat_graph(
        router=ChatRouter(llm=generation.llm),
        knowledge=KnowledgeFlow(
            knowledge,
            generation,
            model_used=model_used or settings.generation_model,
        ),
        lead=LeadService(lead_tool),
        support=SupportService(ticket_tool),
        generation=generation,
    )
    return ConversationOrchestrator(graph, store, traces)
