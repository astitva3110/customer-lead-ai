from app.services.retrieval.merge import merge_candidates
from app.services.retrieval.models import RetrievalCandidate


def _candidate(chunk_id: str, *, vector: float | None = None, keyword: float | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=chunk_id,
        metadata={"page_number": 1},
        vector_score=vector,
        keyword_score=keyword,
    )


def test_merge_deduplicates_by_chunk_id_and_keeps_both_scores() -> None:
    vector = [
        _candidate("A", vector=0.9),
        _candidate("B", vector=0.8),
        _candidate("C", vector=0.7),
    ]
    keyword = [
        _candidate("B", keyword=0.6),
        _candidate("C", keyword=0.5),
        _candidate("D", keyword=0.4),
    ]
    merged = merge_candidates(vector, keyword)
    assert [item.chunk_id for item in merged] == ["A", "B", "C", "D"]
    by_id = {item.chunk_id: item for item in merged}
    assert by_id["A"].vector_score == 0.9
    assert by_id["A"].keyword_score is None
    assert by_id["B"].vector_score == 0.8
    assert by_id["B"].keyword_score == 0.6
    assert by_id["C"].vector_score == 0.7
    assert by_id["C"].keyword_score == 0.5
    assert by_id["D"].vector_score is None
    assert by_id["D"].keyword_score == 0.4
