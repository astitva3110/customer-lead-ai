from __future__ import annotations

from app.services.retrieval.models import RetrievalCandidate
from app.evaluation.fusion import FusionConfig, current_merge, fuse, rank_normalized_fusion, rrf_fusion, weighted_fusion


def _c(chunk_id: str, *, vector: float | None = None, keyword: float | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc",
        text=chunk_id,
        metadata={"section_path": [chunk_id]},
        vector_score=vector,
        keyword_score=keyword,
    )


def test_current_merge_is_vector_first_until_cap():
    vector = [_c("A", vector=0.5), _c("B", vector=0.4), _c("C", vector=0.3)]
    keyword = [_c("D", keyword=0.9), _c("B", keyword=0.2)]
    merged = current_merge(vector, keyword, FusionConfig(top_k=10, cap=30))
    assert [item.chunk_id for item in merged] == ["A", "B", "C", "D"]
    assert merged[1].keyword_score == 0.2


def test_current_merge_cap_sorts_by_max_score():
    vector = [_c(f"v{i}", vector=0.1) for i in range(20)]
    keyword = [_c(f"k{i}", keyword=0.9) for i in range(20)]
    merged = current_merge(vector, keyword, FusionConfig(top_k=10, cap=30))
    assert all(item.chunk_id.startswith("k") for item in merged[:10])


def test_rrf_promotes_overlap():
    vector = [_c("target", vector=0.4), _c("noise", vector=0.9)]
    keyword = [_c("target", keyword=0.3), _c("other", keyword=0.8)]
    fused = rrf_fusion(vector, keyword, FusionConfig(top_k=5, rrf_k=60))
    assert fused[0].chunk_id == "target"


def test_weighted_uses_minmax_not_raw_incomparable_scores():
    vector = [_c("vec_only", vector=0.9), _c("both", vector=0.8)]
    keyword = [_c("kw_only", keyword=2.0), _c("both", keyword=1.8)]
    fused = weighted_fusion(vector, keyword, FusionConfig(top_k=3, weighted_vector=0.7, weighted_keyword=0.3))
    ids = [item.chunk_id for item in fused]
    assert "both" in ids


def test_rank_normalized_ignores_raw_score_scale():
    vector = [_c("A", vector=0.2), _c("B", vector=0.19)]
    keyword = [_c("B", keyword=9.0), _c("C", keyword=8.0)]
    fused = rank_normalized_fusion(vector, keyword, FusionConfig(top_k=3, rank_vector=0.5, rank_keyword=0.5))
    assert fused[0].chunk_id == "B"


def test_fuse_dispatches_all_strategies():
    vector = [_c("A", vector=0.9)]
    keyword = [_c("B", keyword=0.4)]
    config = FusionConfig(top_k=5)
    for name in ("current_merge", "rrf", "weighted", "rank_normalized"):
        rows = fuse(name, vector, keyword, config)
        assert {item.chunk_id for item in rows} == {"A", "B"}
