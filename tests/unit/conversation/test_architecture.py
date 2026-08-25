from pathlib import Path

from llama_index.core.retrievers import BaseRetriever

from app.services.conversation.models import ChatMode
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.providers.knowledge.knowledge_service import LlamaIndexKnowledgeService
from app.providers.knowledge.llama_retriever import HybridLlamaRetriever
from app.providers.reranker.passthrough import PassthroughReranker
from tests.unit.conversation.fakes import make_orchestrator

APP_ROOT = Path(__file__).resolve().parents[3] / "app"


def test_langgraph_owns_workflow() -> None:
    source = (APP_ROOT / "graph" / "chat_graph.py").read_text(encoding="utf-8")
    assert "from langgraph.graph import" in source
    assert "Phase12VectorStore" not in source
    assert "HybridRetriever" not in source
    assert "llama_index" not in source


def test_llamaindex_owns_rag_adapter() -> None:
    source = (APP_ROOT / "providers" / "knowledge" / "llama_retriever.py").read_text(encoding="utf-8")
    assert "from llama_index.core.retrievers import BaseRetriever" in source
    assert "HybridRetriever" in source
    assert issubclass(HybridLlamaRetriever, BaseRetriever)


def test_qwen_generation_only_on_knowledge_path() -> None:
    orchestrator, knowledge, llm, *_ = make_orchestrator()
    orchestrator.handle("arch-1", "I want to buy Radius M16.")
    assert knowledge.queries == []
    assert llm.last_temperature == 0.4
    knowledge_turn = orchestrator.handle("arch-2", "What is Radius M16?")
    assert knowledge.queries[-1] == "What is Radius M16?"
    assert knowledge_turn.sources
    assert llm.call_count >= 2
    assert llm.last_temperature == 0.0


def test_mcp_adapters_are_called_from_services_not_graph() -> None:
    graph_src = (APP_ROOT / "graph" / "chat_graph.py").read_text(encoding="utf-8")
    assert "self._tool.create_lead" not in graph_src
    assert "create_support_ticket" not in graph_src
    lead_src = (APP_ROOT / "services" / "conversation" / "lead_service.py").read_text(encoding="utf-8")
    assert "self._tool.create_lead" in lead_src


def test_no_langchain_orchestration_in_app() -> None:
    for path in APP_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from langchain" not in text
        assert "import langchain" not in text


def test_native_hybrid_knowledge_service_normalizes_hybrid_output() -> None:
    class _Retriever:
        def __init__(self, items: list[RetrievalCandidate]) -> None:
            self.items = items

        def retrieve(self, query: str, *, top_k: int, document_id: str | None = None):
            del query, document_id
            return self.items[:top_k]

    hybrid = HybridRetriever(
        _Retriever(
            [RetrievalCandidate(chunk_id="c1", document_id="d1", text="TINY is compact.", vector_score=0.9)]
        ),
        _Retriever(
            [RetrievalCandidate(chunk_id="c1", document_id="d1", text="TINY is compact.", keyword_score=0.8)]
        ),
        PassthroughReranker(),
        vector_k=20,
        keyword_k=20,
        final_k=6,
    )
    from app.providers.knowledge.hybrid_knowledge_service import HybridKnowledgeService

    service = HybridKnowledgeService(hybrid, retrieval_version="v2")
    result = service.retrieve_knowledge("What is TINY?")
    assert result["query"] == "What is TINY?"
    assert result["retrieval_version"] == "v2"
    assert result["backend"] == "native-hybrid"
    assert result["chunks"][0]["chunk_id"] == "c1"
    assert result["chunks"][0]["text"]
    assert result["retrieval_preview"]["vector"]
    assert result["timings"]
    assert "scores" in result
    assert "metadata" in result


def test_llamaindex_knowledge_service_normalizes_hybrid_output() -> None:
    class _Retriever:
        def __init__(self, items: list[RetrievalCandidate]) -> None:
            self.items = items

        def retrieve(self, query: str, *, top_k: int, document_id: str | None = None):
            del query, document_id
            return self.items[:top_k]

    hybrid = HybridRetriever(
        _Retriever(
            [RetrievalCandidate(chunk_id="c1", document_id="d1", text="TINY is compact.", vector_score=0.9)]
        ),
        _Retriever(
            [RetrievalCandidate(chunk_id="c1", document_id="d1", text="TINY is compact.", keyword_score=0.8)]
        ),
        PassthroughReranker(),
        vector_k=20,
        keyword_k=20,
        final_k=6,
    )
    service = LlamaIndexKnowledgeService(hybrid, retrieval_version="v2")
    result = service.retrieve_knowledge("What is TINY?")
    assert result["query"] == "What is TINY?"
    assert result["retrieval_version"] == "v2"
    assert result["chunks"][0]["chunk_id"] == "c1"
    assert result["chunks"][0]["text"]
    assert result["retrieval_preview"]["vector"]
    assert result["timings"]
    assert "scores" in result
    assert "metadata" in result


def test_state_persistence_round_trip() -> None:
    orchestrator, *_, store = make_orchestrator()
    first = orchestrator.handle("persist-1", "I want to buy Radius M16.")
    loaded = store.get("persist-1")
    assert loaded is not None
    assert loaded.mode == ChatMode.LEAD
    assert loaded.product == first.product
    assert loaded.lead_intent is True
    assert loaded.awaiting_field == ""
    assert "please provide your phone number" not in (first.response or "").lower()
