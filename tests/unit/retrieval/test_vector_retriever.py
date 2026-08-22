from __future__ import annotations

from app.services.retrieval.models import RetrievalCandidate
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.models import RankedHit


class _FakeStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def search(self, query_embedding, *, top_k: int, embedding_version=None, document_id=None):
        self.calls.append(
            {
                "top_k": top_k,
                "embedding_version": embedding_version,
                "document_id": document_id,
                "query_embedding": query_embedding,
            }
        )
        rows = self.rows
        if document_id:
            rows = [row for row in rows if row.get("document_id") == document_id]
        return rows[:top_k]


class _FakeSession:
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        assert queries
        return [[0.1, 0.2, 0.3]]


class _FakeService:
    def __init__(self, hits: list[RankedHit]) -> None:
        self.hits = hits

    def search(self, query: str, *, top_k: int | None = None) -> list[RankedHit]:
        return self.hits[: top_k or len(self.hits)]


def _row(chunk_id: str, score: float, document_id: str = "doc-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": f"text for {chunk_id}",
        "similarity": score,
        "title": "Pricelist",
        "section_path": ["Products"],
        "page_number": 3,
        "split_method": "paragraph",
    }


def test_vector_retriever_returns_expected_ids_and_scores() -> None:
    store = _FakeStore(
        [
            _row("chunk_a", 0.91),
            _row("chunk_b", 0.84),
            _row("chunk_c", 0.70),
        ]
    )
    retriever = VectorCandidateRetriever(
        store=store,
        session=_FakeSession(),
        embedding_version="test-version",
        default_k=20,
    )
    results = retriever.retrieve("battery size", top_k=2)
    assert [item.chunk_id for item in results] == ["chunk_a", "chunk_b"]
    assert results[0].vector_score == 0.91
    assert results[1].vector_score == 0.84
    assert results[0].text == "text for chunk_a"
    assert results[0].metadata["page_number"] == 3
    assert store.calls[0]["top_k"] == 2
    assert store.calls[0]["embedding_version"] == "test-version"


def test_vector_retriever_respects_top_k_via_search_service() -> None:
    hits = [
        RankedHit(
            rank=1,
            similarity=0.88,
            chunk_id="c1",
            document_id="doc-1",
            section_path=["A"],
            token_count=4,
            content_type="paragraph",
            text="one",
        ),
        RankedHit(
            rank=2,
            similarity=0.77,
            chunk_id="c2",
            document_id="doc-1",
            section_path=["A"],
            token_count=4,
            content_type="paragraph",
            text="two",
        ),
        RankedHit(
            rank=3,
            similarity=0.66,
            chunk_id="c3",
            document_id="doc-1",
            section_path=["A"],
            token_count=4,
            content_type="paragraph",
            text="three",
        ),
    ]
    retriever = VectorCandidateRetriever(search_service=_FakeService(hits), default_k=20)
    results = retriever.retrieve("query", top_k=2)
    assert [item.chunk_id for item in results] == ["c1", "c2"]
    assert all(isinstance(item, RetrievalCandidate) for item in results)
    assert results[0].vector_score == 0.88
