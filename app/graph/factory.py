from __future__ import annotations

from app.config import settings
from app.providers.llm.factory import get_semantic_router_provider
from app.services.conversation.knowledge_flow import KnowledgeFlow
from app.services.conversation.lead_service import LeadService
from app.services.conversation.orchestrator import ConversationOrchestrator
from app.services.conversation.router import ChatRouter
from app.services.conversation.support_service import SupportService
from app.services.generation.generation_service import GenerationService
from app.interfaces.providers.business import LeadTool, TicketTool
from app.interfaces.providers.crm import CrmLeadPort
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
    crm: CrmLeadPort | None = None,
) -> ConversationOrchestrator:
    graph = build_chat_graph(
        router=ChatRouter(router_llm=get_semantic_router_provider(settings)),
        knowledge=KnowledgeFlow(
            knowledge,
            generation,
            model_used=model_used or settings.generation_model,
        ),
        lead=LeadService(lead_tool, crm=crm),
        support=SupportService(ticket_tool, crm=crm),
        generation=generation,
    )
    return ConversationOrchestrator(graph, store, traces)
