from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate


class HybridLlamaRetriever(BaseRetriever):
    """LlamaIndex retriever that delegates to the existing hybrid pipeline."""

    def __init__(self, hybrid: HybridRetriever) -> None:
        super().__init__()
        self._hybrid = hybrid
        self.last_result = None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = self._hybrid.search_detailed(query_bundle.query_str)
        self.last_result = result
        return [_node_from_candidate(candidate) for candidate in result.final_candidates]


def _node_from_candidate(candidate: RetrievalCandidate) -> NodeWithScore:
    node = TextNode(
        text=candidate.text,
        id_=candidate.chunk_id,
        metadata={
            "chunk_id": candidate.chunk_id,
            "document_id": candidate.document_id,
            "section_path": candidate.section_path,
            "page_number": candidate.page_number,
            "title": candidate.document_title,
            "content_type": candidate.content_type,
            "knowledge_key": (candidate.metadata or {}).get("knowledge_key"),
            "chunking_algorithm_version": (candidate.metadata or {}).get("chunking_algorithm_version"),
            "embedding_input_manifest": (candidate.metadata or {}).get("embedding_input_manifest"),
        },
    )
    score = candidate.rerank_score
    if score is None:
        score = candidate.vector_score
    return NodeWithScore(node=node, score=float(score or 0.0))
