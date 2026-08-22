"""Phase 16 in-memory V2 vs V3 retrieval inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.phase13_corpus import load_phase13_validated_corpus
from app.kb.evaluation.phase15_kb_v2_dataset import PHASE15_QUESTIONS, resolve_expected_knowledge
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.phase16_corpus import Phase16Corpus, load_phase16_v3_corpus


@dataclass
class InMemoryRankResult:
    chunk_id: str
    rank: int
    similarity: float
    section_path: list[str]
    token_count: int
    text: str


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _rank_chunks(
    query_vector: list[float],
    chunks: list[Phase12ChunkRecord],
    *,
    embed_texts: list[list[float]],
    top_k: int = 100,
) -> list[InMemoryRankResult]:
    scored: list[tuple[float, Phase12ChunkRecord]] = []
    for chunk, vector in zip(chunks, embed_texts, strict=True):
        scored.append((_cosine_similarity(query_vector, vector), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[InMemoryRankResult] = []
    for index, (similarity, chunk) in enumerate(scored[:top_k], start=1):
        results.append(
            InMemoryRankResult(
                chunk_id=chunk.chunk_id,
                rank=index,
                similarity=round(similarity, 6),
                section_path=list(chunk.section_path),
                token_count=chunk.token_count,
                text=chunk.content,
            )
        )
    return results


def _find_expected_in_corpus(
    corpus_chunks: list[Phase12ChunkRecord],
    expected_chunk_ids: list[str],
    patterns: list[str],
) -> Phase12ChunkRecord | None:
    id_map = {chunk.chunk_id: chunk for chunk in corpus_chunks}
    for chunk_id in expected_chunk_ids:
        if chunk_id in id_map:
            return id_map[chunk_id]
    for chunk in corpus_chunks:
        haystack = f"{chunk.content} {' '.join(chunk.section_path)}".lower()
        if any(pattern.lower() in haystack for pattern in patterns):
            return chunk
    return None


QUERY_PATTERNS: dict[str, list[str]] = {
    "ERK-V2-001": ["behind-the-ear (bte)", "behind-the-ear", "bte"],
    "ERK-V2-002": ["return and refund", "returns, replacements"],
    "ERK-V2-003": ["registered office", "corporate office"],
    "ERK-V2-004": ["customer support", "contact", "reach us", "phone", "email"],
    "ERK-V2-005": ["delivery", "deliver", "dispatch", "shipping"],
    "ERK-V2-006": ["registered office", "corporate office"],
    "ERK-V2-007": ["omni"],
    "ERK-V2-008": ["battery life", "battery lasts", "hours of use"],
    "ERK-V2-009": ["why choose earkart", "why choose earKART", "what makes us better"],
}


def run_phase16_inspection(*, device: str | None = None) -> dict[str, Any]:
    if device:
        import os

        os.environ["EMBEDDING_DEVICE"] = device

    from dataclasses import replace

    config = replace(EmbeddingConfig.from_settings(), device=device or settings.embedding_device)
    provider = create_embedding_provider(config)

    v2_corpus = load_phase13_validated_corpus()
    v3_corpus = load_phase16_v3_corpus()
    expected_map = resolve_expected_knowledge(v2_corpus)

    v2_chunks = v2_corpus.all_chunks
    v3_chunks = v3_corpus.all_chunks

    v2_doc_vectors = provider.embed_documents([chunk.embedding_input for chunk in v2_chunks])
    v3_doc_vectors = provider.embed_documents([chunk.embedding_input for chunk in v3_chunks])

    inspections: list[dict[str, Any]] = []
    focus_ids = {"ERK-V2-001", "ERK-V2-004", "ERK-V2-007", "ERK-V2-009"}

    for query_id, question in PHASE15_QUESTIONS:
        query_vector = provider.embed_queries([question])[0]
        v2_ranked = _rank_chunks(query_vector, v2_chunks, embed_texts=v2_doc_vectors, top_k=100)
        v3_ranked = _rank_chunks(query_vector, v3_chunks, embed_texts=v3_doc_vectors, top_k=100)

        expected = expected_map[query_id]
        patterns = QUERY_PATTERNS.get(query_id, [])
        v2_expected = _find_expected_in_corpus(v2_chunks, expected.expected_chunk_ids, patterns)
        v3_expected = _find_expected_in_corpus(v3_chunks, [], patterns)

        v2_expected_rank = next(
            (hit.rank for hit in v2_ranked if v2_expected and hit.chunk_id == v2_expected.chunk_id),
            None,
        )
        v3_expected_rank = next(
            (hit.rank for hit in v3_ranked if v3_expected and hit.chunk_id == v3_expected.chunk_id),
            None,
        )
        v2_expected_sim = next(
            (hit.similarity for hit in v2_ranked if v2_expected and hit.chunk_id == v2_expected.chunk_id),
            None,
        )
        v3_expected_sim = next(
            (hit.similarity for hit in v3_ranked if v3_expected and hit.chunk_id == v3_expected.chunk_id),
            None,
        )

        inspections.append(
            {
                "id": query_id,
                "query": question,
                "answerability": expected.answerability,
                "expected_v2_chunk_id": v2_expected.chunk_id if v2_expected else None,
                "expected_v3_chunk_id": v3_expected.chunk_id if v3_expected else None,
                "expected_section_path": v3_expected.section_path if v3_expected else expected.expected_section_path,
                "v2_expected_rank": v2_expected_rank,
                "v3_expected_rank": v3_expected_rank,
                "v2_expected_similarity": v2_expected_sim,
                "v3_expected_similarity": v3_expected_sim,
                "rank_delta_v2_to_v3": (
                    (v2_expected_rank - v3_expected_rank)
                    if v2_expected_rank is not None and v3_expected_rank is not None
                    else None
                ),
                "improved": (
                    v3_expected_rank is not None
                    and v2_expected_rank is not None
                    and v3_expected_rank < v2_expected_rank
                ),
                "v3_top10": [
                    {
                        "rank": hit.rank,
                        "similarity": hit.similarity,
                        "chunk_id": hit.chunk_id,
                        "section_path": hit.section_path,
                        "token_count": hit.token_count,
                        "text": hit.text[:300],
                    }
                    for hit in v3_ranked[:10]
                ],
                "focus_query": query_id in focus_ids,
            }
        )

    focus = [item for item in inspections if item["focus_query"]]
    improved_count = sum(1 for item in inspections if item.get("improved"))
    v3_top5_wins = sum(
        1
        for item in inspections
        if item.get("v3_expected_rank") is not None and item["v3_expected_rank"] <= 5
    )

    return {
        "inspection_mode": "in_memory_cosine",
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "pgvector_writes": 0,
        "v2_chunk_count": len(v2_chunks),
        "v3_chunk_count": len(v3_chunks),
        "improved_queries": improved_count,
        "v3_expected_in_top5": v3_top5_wins,
        "focus_comparison": focus,
        "queries": inspections,
    }
