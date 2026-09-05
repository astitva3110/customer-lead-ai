from dataclasses import replace

from app.services.retrieval.models import RetrievalCandidate
from app.providers.reranker.lexical import LexicalOverlapReranker
from app.providers.reranker.passthrough import PassthroughReranker


class _ScoreMapReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def rerank(self, query: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        del query
        ranked = [replace(item, rerank_score=self.scores[item.chunk_id]) for item in candidates]
        ranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return ranked


def _candidate(chunk_id: str, text: str = "") -> RetrievalCandidate:
    return RetrievalCandidate(chunk_id=chunk_id, document_id="doc-1", text=text or chunk_id)


def test_mock_reranker_orders_by_score() -> None:
    reranker = _ScoreMapReranker({"A": 0.91, "B": 0.82, "C": 0.40})
    ranked = reranker.rerank("q", [_candidate("C"), _candidate("A"), _candidate("B")])
    assert [item.chunk_id for item in ranked] == ["A", "B", "C"]
    assert [item.rerank_score for item in ranked] == [0.91, 0.82, 0.40]


def test_passthrough_reranker_uses_existing_scores() -> None:
    ranked = PassthroughReranker().rerank(
        "q",
        [
            RetrievalCandidate(chunk_id="B", document_id="d", text="b", vector_score=0.82),
            RetrievalCandidate(chunk_id="A", document_id="d", text="a", vector_score=0.91),
            RetrievalCandidate(chunk_id="C", document_id="d", text="c", keyword_score=0.40),
        ],
    )
    assert [item.chunk_id for item in ranked] == ["A", "B", "C"]


def test_lexical_overlap_reranker_prefers_term_matches() -> None:
    ranked = LexicalOverlapReranker().rerank(
        "Radius 16 MRP",
        [
            _candidate("other", "FAME series zinc air battery"),
            _candidate("price", "Radius 16 — 16 Channels — MRP: 27500"),
        ],
    )
    assert ranked[0].chunk_id == "price"
    assert ranked[0].rerank_score is not None
    assert len(ranked) == 1


def test_lexical_reranker_keeps_type_chunk_above_faq_boilerplate() -> None:
    faqs = [
        RetrievalCandidate(
            chunk_id=f"faq-{index}",
            document_id="faq",
            text="How do I know if I have hearing loss? Many people first notice they need a hearing aid.",
            vector_score=0.40,
        )
        for index in range(6)
    ]
    type_chunk = RetrievalCandidate(
        chunk_id="type-cic",
        document_id="catalog",
        text="5.3 Compact CIC — Rechargeable CIC Hearing Aid (Specifications)",
        metadata={"document_title": "Product catalogue", "section_path": ["Products", "CIC"]},
        vector_score=0.48,
    )
    ranked = LexicalOverlapReranker().rerank(
        "hi how are u ? i whant to know about CIC",
        [*faqs, type_chunk],
    )
    assert ranked[0].chunk_id == "type-cic"
