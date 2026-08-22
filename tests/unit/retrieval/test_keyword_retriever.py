from __future__ import annotations

from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever


class _FakeKeywordStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def ensure_content_fts_index(self) -> None:
        return None

    def search_keyword(
        self,
        query: str,
        *,
        top_k: int,
        embedding_version=None,
        document_id=None,
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "embedding_version": embedding_version,
                "document_id": document_id,
            }
        )
        rows = [row for row in self.rows if query.lower() in (row.get("content") or "").lower()]
        if document_id:
            rows = [row for row in rows if row.get("document_id") == document_id]
        return rows[:top_k]


def test_keyword_retriever_returns_matches_with_scores() -> None:
    store = _FakeKeywordStore(
        [
            {
                "chunk_id": "kw_a",
                "document_id": "doc-1",
                "content": "Radius 16 MRP 27500",
                "keyword_score": 0.42,
                "title": "Radius",
                "section_path": ["5.2 Radius"],
                "page_number": 3,
            },
            {
                "chunk_id": "kw_b",
                "document_id": "doc-1",
                "content": "TINY rechargeable CIC",
                "keyword_score": 0.11,
                "title": "TINY",
                "section_path": ["5.3 TINY"],
                "page_number": 3,
            },
            {
                "chunk_id": "kw_c",
                "document_id": "doc-1",
                "content": "Radius 8 MRP 22500",
                "keyword_score": 0.31,
                "title": "Radius",
                "section_path": ["5.2 Radius"],
                "page_number": 3,
            },
        ]
    )
    retriever = KeywordCandidateRetriever(store, embedding_version="test-version", default_k=20)
    results = retriever.retrieve("Radius", top_k=2)
    assert [item.chunk_id for item in results] == ["kw_a", "kw_c"]
    assert results[0].keyword_score == 0.42
    assert results[1].keyword_score == 0.31
    assert all(item.vector_score is None for item in results)
    assert store.calls[0]["top_k"] == 2
    assert store.calls[0]["query"] == "Radius"


def test_keyword_retriever_respects_top_k() -> None:
    store = _FakeKeywordStore(
        [
            {
                "chunk_id": f"kw_{index}",
                "document_id": "doc-1",
                "content": "ISO 13485 certificate",
                "keyword_score": 1.0 - index * 0.1,
                "title": "ISO",
                "section_path": [],
                "page_number": 1,
            }
            for index in range(5)
        ]
    )
    retriever = KeywordCandidateRetriever(store, ensure_index=False)
    results = retriever.retrieve("ISO 13485", top_k=3)
    assert len(results) == 3
    assert [item.chunk_id for item in results] == ["kw_0", "kw_1", "kw_2"]
