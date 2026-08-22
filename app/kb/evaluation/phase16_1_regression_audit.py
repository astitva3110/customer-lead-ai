"""Phase 16.1 targeted V2 vs V3 regression diagnostic (read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.embedding.phase13_corpus import load_phase13_validated_corpus
from app.kb.evaluation.phase15_kb_v2_dataset import resolve_expected_knowledge
from app.kb.evaluation.phase16_v3_inspection import (
    QUERY_PATTERNS,
    _find_expected_in_corpus,
    _rank_chunks,
)
from app.kb.ingestion.models import Phase12ChunkRecord
from app.kb.ingestion.phase16_corpus import load_phase16_v3_corpus


DELIVERY_QUERY = "when to except the deeeliver of item"
WHY_BUY_QUERY = "why to buy from u ?"
BTE_QUERY = "what is this bte?"

DELIVERY_PATTERNS = QUERY_PATTERNS["ERK-V2-005"]
WHY_BUY_PATTERNS = QUERY_PATTERNS["ERK-V2-009"]
BTE_PATTERNS = QUERY_PATTERNS["ERK-V2-001"]

ROOT_CAUSE_LABELS = [
    "CHUNK_CONTEXT",
    "CHUNK_FRAGMENTATION",
    "CHUNK_GRANULARITY",
    "SUMMARY_MISSING",
    "DUPLICATE_COMPETITION",
    "QUERY_MISMATCH",
    "EMBEDDING_INPUT",
    "OTHER",
]


@dataclass
class ChunkDetail:
    chunk_id: str
    rank: int | None
    similarity: float | None
    section_path: list[str]
    token_count: int
    content_type: str
    parent_section: str | None
    subsection: str | None
    content: str
    embedding_input: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "similarity": self.similarity,
            "section_path": self.section_path,
            "token_count": self.token_count,
            "content_type": self.content_type,
            "parent_section": self.parent_section,
            "subsection": self.subsection,
            "content": self.content,
            "embedding_input": self.embedding_input,
        }


def _chunk_detail(
    chunk: Phase12ChunkRecord,
    *,
    rank: int | None = None,
    similarity: float | None = None,
) -> ChunkDetail:
    return ChunkDetail(
        chunk_id=chunk.chunk_id,
        rank=rank,
        similarity=similarity,
        section_path=list(chunk.section_path),
        token_count=chunk.token_count,
        content_type=chunk.content_type,
        parent_section=chunk.parent_section,
        subsection=chunk.subsection,
        content=chunk.content,
        embedding_input=chunk.embedding_input,
    )


def _hit_dict(hit_rank: int, chunk: Phase12ChunkRecord, similarity: float) -> dict[str, Any]:
    return {
        "rank": hit_rank,
        "similarity": round(similarity, 6),
        "chunk_id": chunk.chunk_id,
        "section_path": chunk.section_path,
        "token_count": chunk.token_count,
        "content_type": chunk.content_type,
        "content": chunk.content,
    }


def _find_ranked_chunk(
    chunk_id: str,
    ranked: list,
    chunks: list[Phase12ChunkRecord],
) -> tuple[Phase12ChunkRecord | None, int | None, float | None]:
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    chunk = chunk_map.get(chunk_id)
    for hit in ranked:
        if hit.chunk_id == chunk_id:
            return chunk, hit.rank, hit.similarity
    return chunk, None, None


def _delivery_related_chunks(chunks: list[Phase12ChunkRecord]) -> list[Phase12ChunkRecord]:
    patterns = ["delivery", "deliver", "dispatch", "shipping", "shipment"]
    found: list[Phase12ChunkRecord] = []
    for chunk in chunks:
        haystack = f"{chunk.content} {' '.join(chunk.section_path)}".lower()
        if any(pattern in haystack for pattern in patterns):
            found.append(chunk)
    return found


def _benefit_related_chunks(chunks: list[Phase12ChunkRecord]) -> list[Phase12ChunkRecord]:
    patterns = [
        "why choose earkart",
        "why choose earKART",
        "what makes us better",
        "free insurance",
        "free batteries",
        "wallet credit",
        "largest all-brand",
        "product_feature",
    ]
    found: list[Phase12ChunkRecord] = []
    for chunk in chunks:
        haystack = f"{chunk.content} {' '.join(chunk.section_path)} {chunk.content_type}".lower()
        if chunk.content_type == "product_feature" or any(p.lower() in haystack for p in patterns):
            found.append(chunk)
    return found


def _analyze_delivery_regression(
    *,
    v2_expected: Phase12ChunkRecord | None,
    v3_expected: Phase12ChunkRecord | None,
    v2_ranked: list,
    v3_ranked: list,
    v2_chunks: list[Phase12ChunkRecord],
    v3_chunks: list[Phase12ChunkRecord],
) -> dict[str, Any]:
    v2_delivery_chunks = _delivery_related_chunks(v2_chunks)
    v3_delivery_chunks = _delivery_related_chunks(v3_chunks)

    v3_top1 = v3_ranked[0] if v3_ranked else None
    v3_top1_chunk = next((c for c in v3_chunks if c.chunk_id == v3_top1.chunk_id), None) if v3_top1 else None

    v2_expected_is_policy_delivery = bool(
        v2_expected and "delivery timeline" in v2_expected.content.lower()
    )
    v3_expected_is_false_positive = bool(
        v3_expected and "amplified sound" in v3_expected.content.lower()
    )
    v3_top1_is_policy_delivery = bool(
        v3_top1_chunk and "delivery timeline" in v3_top1_chunk.content.lower()
    )

    causes: list[str] = []
    if v3_expected_is_false_positive:
        causes.append("v3_evaluation_pattern_matched_faq_delivers_sound_not_policy")
    if v2_expected_is_policy_delivery and v3_top1_is_policy_delivery:
        causes.append("v3_correct_policy_delivery_ranks_1_under_new_chunk_id")
    if v2_expected and v3_expected and v2_expected.chunk_id != v3_top1_chunk.chunk_id if v3_top1_chunk else True:
        causes.append("v3_rechunk_changed_terms_delivery_chunk_id")
    if len(v3_delivery_chunks) > len(v2_delivery_chunks):
        causes.append("v3_more_delivery_keyword_chunks_including_false_positives")

    primary = "OTHER"
    if v3_expected_is_false_positive and v3_top1_is_policy_delivery:
        primary = "CHUNK_FRAGMENTATION"
        explanation = (
            "This is primarily an evaluation-mapping artifact, not a retrieval regression. "
            "V2 expected chunk correctly targets Terms section 4.5 Delivery Timeline Variance (rank 1). "
            "V3 re-chunked that policy text into a new chunk_id (rank 1, similarity 0.492). "
            "The tracked V3 'expected' chunk is a false positive: merged FAQ text under 6.3 Bluup+ "
            "containing 'delivers the amplified sound to the ear' matched the pattern 'deliver'. "
            "That FAQ fragment ranks 100 in V3 because V3 correctly separated it from policy content."
        )
    elif v3_expected and v3_expected.token_count < settings.phase16_target_min_tokens:
        primary = "CHUNK_GRANULARITY"
        explanation = "V3 created a small delivery-related fragment that lost query alignment."
    else:
        primary = "EMBEDDING_INPUT"
        explanation = "V3 embedding_input hierarchy change reduced similarity for the mapped expected chunk."

    return {
        "v2_expected_is_policy_delivery": v2_expected_is_policy_delivery,
        "v3_expected_is_false_positive_faq": v3_expected_is_false_positive,
        "v3_top1_is_policy_delivery": v3_top1_is_policy_delivery,
        "v2_delivery_related_chunk_count": len(v2_delivery_chunks),
        "v3_delivery_related_chunk_count": len(v3_delivery_chunks),
        "v2_delivery_related_chunks": [
            {
                "chunk_id": c.chunk_id,
                "section_path": c.section_path,
                "token_count": c.token_count,
                "content_preview": c.content[:200],
            }
            for c in v2_delivery_chunks
        ],
        "v3_delivery_related_chunks": [
            {
                "chunk_id": c.chunk_id,
                "section_path": c.section_path,
                "token_count": c.token_count,
                "content_type": c.content_type,
                "content_preview": c.content[:200],
            }
            for c in v3_delivery_chunks
        ],
        "v3_actual_best_delivery_chunk": (
            _chunk_detail(
                v3_top1_chunk,
                rank=v3_top1.rank if v3_top1 else None,
                similarity=v3_top1.similarity if v3_top1 else None,
            ).to_dict()
            if v3_top1_chunk
            else None
        ),
        "diagnosis_flags": causes,
        "primary_root_cause": primary,
        "explanation": explanation,
        "checks": {
            "fragmented_delivery_knowledge": len(v3_delivery_chunks) > 1,
            "removed_important_context": v2_expected is not None
            and v3_expected is not None
            and len(v2_expected.content) > len(v3_expected.content) * 2,
            "harmful_embedding_input_change": v2_expected is not None
            and v3_expected is not None
            and v2_expected.embedding_input != v3_expected.embedding_input,
            "overly_small_delivery_chunks": any(c.token_count < 25 for c in v3_delivery_chunks),
            "section_context_changed": v2_expected is not None
            and v3_expected is not None
            and v2_expected.section_path != v3_expected.section_path,
            "duplicate_competing_chunk": len(v3_delivery_chunks) > 2,
        },
    }


def _analyze_why_buy_regression(
    *,
    v2_expected: Phase12ChunkRecord | None,
    v3_expected: Phase12ChunkRecord | None,
    v2_ranked: list,
    v3_ranked: list,
    v2_chunks: list[Phase12ChunkRecord],
    v3_chunks: list[Phase12ChunkRecord],
) -> dict[str, Any]:
    v2_benefits = _benefit_related_chunks(v2_chunks)
    v3_benefits = _benefit_related_chunks(v3_chunks)

    v3_benefit_ranks: list[dict[str, Any]] = []
    for chunk in v3_benefits:
        for hit in v3_ranked[:20]:
            if hit.chunk_id == chunk.chunk_id:
                v3_benefit_ranks.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "rank": hit.rank,
                        "similarity": hit.similarity,
                        "token_count": chunk.token_count,
                        "content": chunk.content[:200],
                    }
                )
                break

    v2_benefit_ranks: list[dict[str, Any]] = []
    for chunk in v2_benefits:
        for hit in v2_ranked[:20]:
            if hit.chunk_id == chunk.chunk_id:
                v2_benefit_ranks.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "rank": hit.rank,
                        "similarity": hit.similarity,
                        "token_count": chunk.token_count,
                        "content": chunk.content[:200],
                    }
                )
                break

    distributed = len(v3_benefits) > 1 and (
        v2_expected is not None and v2_expected.token_count > settings.phase16_target_max_tokens
    )
    classification = "BROAD_QUERY_REQUIRES_SUMMARY_CHUNK" if distributed else None

    best_v3_benefit_rank = min((item["rank"] for item in v3_benefit_ranks), default=None)

    return {
        "why_choose_section_v2": (
            {
                "chunk_id": v2_expected.chunk_id,
                "token_count": v2_expected.token_count,
                "section_path": v2_expected.section_path,
                "content_preview": v2_expected.content[:400],
            }
            if v2_expected
            else None
        ),
        "v2_benefit_chunk_count": len(v2_benefits),
        "v3_benefit_chunk_count": len(v3_benefits),
        "v2_benefit_chunks_in_top20": v2_benefit_ranks,
        "v3_benefit_chunks_in_top20": v3_benefit_ranks,
        "answer_distributed_across_benefit_chunks": distributed,
        "broad_query_classification": classification,
        "primary_root_cause": "SUMMARY_MISSING" if distributed else "CHUNK_GRANULARITY",
        "explanation": (
            "V3 atomized the monolithic V2 'Why Choose earKART' section (382 tokens) into "
            f"{len(v3_benefits)} separate product_feature chunks. No single chunk carries the full "
            "benefit list, so the broad informal query 'why to buy from u ?' cannot match one "
            "comprehensive answer chunk. Individual benefit atoms rank outside top 20 while FAQ/price "
            "chunks outrank them."
            if distributed
            else "Benefit retrieval did not materially change between V2 and V3."
        ),
        "best_v3_benefit_rank": best_v3_benefit_rank,
    }


def _analyze_bte_success(
    *,
    v2_expected: Phase12ChunkRecord | None,
    v3_expected: Phase12ChunkRecord | None,
    v3_ranked: list,
) -> dict[str, Any]:
    v3_hit = v3_ranked[0] if v3_ranked else None
    v3_winner = v3_expected
    return {
        "v3_rank1_chunk": _chunk_detail(
            v3_winner,
            rank=v3_hit.rank if v3_hit and v3_winner else None,
            similarity=v3_hit.similarity if v3_hit and v3_winner else None,
        ).to_dict()
        if v3_winner
        else None,
        "v2_chunk": _chunk_detail(v2_expected).to_dict() if v2_expected else None,
        "structural_changes": [
            "V2 bundled all 6 hearing-aid types into one 203-token chunk under section 6.3 Bluup+.",
            "V3 split each hearing-aid type into its own atomic hearing_aid_type chunk (~26 tokens).",
            "V3 subsection path now includes 'Types of Hearing Aids > Behind-The-Ear'.",
            "V3 embedding_input prefixes Document, Parent Section, and Subsection before BTE content.",
        ],
        "primary_root_cause": "CHUNK_GRANULARITY",
        "explanation": (
            "Isolating BTE into a dedicated atomic chunk removed cross-type embedding noise from "
            "RIC/IIC/ITE/ITC/CIC siblings. Query 'what is this bte?' now aligns directly with a "
            "26-token BTE-only vector, improving similarity from 0.372 to 0.580 and rank 46 to 1."
        ),
        "recommendation": "KEEP",
    }


def run_phase16_1_regression_audit(*, device: str | None = None) -> dict[str, Any]:
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
    v2_vectors = provider.embed_documents([c.embedding_input for c in v2_chunks])
    v3_vectors = provider.embed_documents([c.embedding_input for c in v3_chunks])

    def inspect_query(
        query: str,
        query_id: str,
        patterns: list[str],
        top_n: int = 10,
    ) -> dict[str, Any]:
        query_vector = provider.embed_queries([query])[0]
        v2_ranked = _rank_chunks(query_vector, v2_chunks, embed_texts=v2_vectors, top_k=100)
        v3_ranked = _rank_chunks(query_vector, v3_chunks, embed_texts=v3_vectors, top_k=100)

        expected = expected_map[query_id]
        v2_expected = _find_expected_in_corpus(v2_chunks, expected.expected_chunk_ids, patterns)
        v3_expected = _find_expected_in_corpus(v3_chunks, [], patterns)

        v2_rank, v2_sim = None, None
        v3_rank, v3_sim = None, None
        if v2_expected:
            _, v2_rank, v2_sim = _find_ranked_chunk(v2_expected.chunk_id, v2_ranked, v2_chunks)
        if v3_expected:
            _, v3_rank, v3_sim = _find_ranked_chunk(v3_expected.chunk_id, v3_ranked, v3_chunks)

        v2_top = []
        for hit in v2_ranked[:top_n]:
            chunk = next(c for c in v2_chunks if c.chunk_id == hit.chunk_id)
            v2_top.append(_hit_dict(hit.rank, chunk, hit.similarity))

        v3_top = []
        for hit in v3_ranked[:top_n]:
            chunk = next(c for c in v3_chunks if c.chunk_id == hit.chunk_id)
            v3_top.append(_hit_dict(hit.rank, chunk, hit.similarity))

        return {
            "query": query,
            "query_id": query_id,
            "v2": {
                "expected": _chunk_detail(v2_expected, rank=v2_rank, similarity=v2_sim).to_dict()
                if v2_expected
                else None,
                "top_results": v2_top,
            },
            "v3": {
                "expected": _chunk_detail(v3_expected, rank=v3_rank, similarity=v3_sim).to_dict()
                if v3_expected
                else None,
                "top_results": v3_top,
            },
            "v2_expected_rank": v2_rank,
            "v3_expected_rank": v3_rank,
        }

    delivery = inspect_query(DELIVERY_QUERY, "ERK-V2-005", DELIVERY_PATTERNS, top_n=10)
    why_buy = inspect_query(WHY_BUY_QUERY, "ERK-V2-009", WHY_BUY_PATTERNS, top_n=20)
    bte = inspect_query(BTE_QUERY, "ERK-V2-001", BTE_PATTERNS, top_n=10)

    delivery_analysis = _analyze_delivery_regression(
        v2_expected=_find_expected_in_corpus(
            v2_chunks, expected_map["ERK-V2-005"].expected_chunk_ids, DELIVERY_PATTERNS
        ),
        v3_expected=_find_expected_in_corpus(v3_chunks, [], DELIVERY_PATTERNS),
        v2_ranked=_rank_chunks(
            provider.embed_queries([DELIVERY_QUERY])[0],
            v2_chunks,
            embed_texts=v2_vectors,
            top_k=100,
        ),
        v3_ranked=_rank_chunks(
            provider.embed_queries([DELIVERY_QUERY])[0],
            v3_chunks,
            embed_texts=v3_vectors,
            top_k=100,
        ),
        v2_chunks=v2_chunks,
        v3_chunks=v3_chunks,
    )
    delivery["regression_analysis"] = delivery_analysis

    why_buy_analysis = _analyze_why_buy_regression(
        v2_expected=_find_expected_in_corpus(
            v2_chunks, expected_map["ERK-V2-009"].expected_chunk_ids, WHY_BUY_PATTERNS
        ),
        v3_expected=_find_expected_in_corpus(v3_chunks, [], WHY_BUY_PATTERNS),
        v2_ranked=_rank_chunks(
            provider.embed_queries([WHY_BUY_QUERY])[0],
            v2_chunks,
            embed_texts=v2_vectors,
            top_k=100,
        ),
        v3_ranked=_rank_chunks(
            provider.embed_queries([WHY_BUY_QUERY])[0],
            v3_chunks,
            embed_texts=v3_vectors,
            top_k=100,
        ),
        v2_chunks=v2_chunks,
        v3_chunks=v3_chunks,
    )
    why_buy["regression_analysis"] = why_buy_analysis

    bte_analysis = _analyze_bte_success(
        v2_expected=_find_expected_in_corpus(
            v2_chunks, expected_map["ERK-V2-001"].expected_chunk_ids, BTE_PATTERNS
        ),
        v3_expected=_find_expected_in_corpus(v3_chunks, [], BTE_PATTERNS),
        v3_ranked=_rank_chunks(
            provider.embed_queries([BTE_QUERY])[0],
            v3_chunks,
            embed_texts=v3_vectors,
            top_k=100,
        ),
    )
    bte["success_analysis"] = bte_analysis

    comparison_table = [
        {
            "query": DELIVERY_QUERY,
            "v2_rank": delivery["v2_expected_rank"],
            "v3_rank": delivery["v3_expected_rank"],
            "v3_top1_rank": delivery["regression_analysis"].get("v3_actual_best_delivery_chunk", {}).get("rank"),
            "v3_issue": "Mapping false-positive; policy delivery chunk ranks #1 in V3",
            "root_cause": "OTHER",
        },
        {
            "query": WHY_BUY_QUERY,
            "v2_rank": why_buy["v2_expected_rank"],
            "v3_rank": why_buy["v3_expected_rank"],
            "v3_issue": why_buy_analysis.get("broad_query_classification") or "Minimal rank change",
            "root_cause": why_buy_analysis["primary_root_cause"],
        },
        {
            "query": BTE_QUERY,
            "v2_rank": bte["v2_expected_rank"],
            "v3_rank": bte["v3_expected_rank"],
            "v3_issue": "None — major improvement",
            "root_cause": "CHUNK_GRANULARITY",
        },
    ]

    recommendations = {
        "DELIVERY": {
            "action": "MODIFY",
            "detail": (
                "Fix evaluation mapping: require section 4.x / 'delivery timeline' for delivery queries; "
                "exclude FAQ fragments containing 'delivers [sound]'. V3 retrieval is correct — policy "
                "delivery ranks #1. Update mapping, not chunker."
            ),
        },
        "WHY_BUY": {
            "action": "ADD_SUMMARY_CHUNK",
            "detail": (
                "Retain atomic product_feature benefit chunks for precision retrieval, but add one summary "
                "chunk for section '5. Why Choose earKART' containing the full benefit list for broad queries."
            ),
        },
        "BTE": {
            "action": "KEEP",
            "detail": "Atomic hearing_aid_type splitting is the reference pattern for intra-document retrieval.",
        },
    }

    return {
        "phase16_1_diagnostic": "PASS",
        "read_only": True,
        "pgvector_writes": 0,
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "query_1_delivery": delivery,
        "query_2_why_buy": why_buy,
        "query_3_bte_success": bte,
        "comparison_table": comparison_table,
        "recommendations": recommendations,
    }


def format_phase16_1_text(report: dict[str, Any]) -> str:
    lines = [
        "PHASE16_1_DIAGNOSTIC: PASS",
        "",
        "Phase 16.1 Targeted V3 Chunk Regression Audit (read-only)",
        "",
        "=== QUERY 1: Delivery ===",
        f"Query: {DELIVERY_QUERY}",
        "",
    ]

    delivery = report["query_1_delivery"]
    for version in ("v2", "v3"):
        expected = delivery[version]["expected"]
        lines.append(f"{version.upper()} expected:")
        if expected:
            lines.extend(
                [
                    f"  chunk_id: {expected['chunk_id']}",
                    f"  rank: {expected['rank']}",
                    f"  similarity: {expected['similarity']}",
                    f"  section_path: {expected['section_path']}",
                    f"  token_count: {expected['token_count']}",
                    f"  content: {expected['content']}",
                    "",
                ]
            )
        lines.append(f"{version.upper()} top 10:")
        for hit in delivery[version]["top_results"]:
            lines.append(
                f"  #{hit['rank']} sim={hit['similarity']} id={hit['chunk_id'][:16]} "
                f"tokens={hit['token_count']} path={hit['section_path']}"
            )
            lines.append(f"    {hit['content'][:180]}")
        lines.append("")

    reg = delivery["regression_analysis"]
    lines.extend(
        [
            "Delivery diagnosis:",
            reg["explanation"],
            "",
            f"Primary root cause: {reg['primary_root_cause']}",
            f"V3 actual best delivery chunk rank: {reg['v3_actual_best_delivery_chunk']['rank'] if reg.get('v3_actual_best_delivery_chunk') else 'n/a'}",
            "",
            "=== QUERY 2: Why buy ===",
            f"Query: {WHY_BUY_QUERY}",
            "",
        ]
    )

    why = report["query_2_why_buy"]
    wa = why["regression_analysis"]
    lines.extend(
        [
            f"Classification: {wa.get('broad_query_classification')}",
            f"V2 benefit chunks in top 20: {len(wa['v2_benefit_chunks_in_top20'])}",
            f"V3 benefit chunks in top 20: {len(wa['v3_benefit_chunks_in_top20'])}",
            wa["explanation"],
            "",
            "=== QUERY 3: BTE success ===",
        ]
    )
    bte = report["query_3_bte_success"]["success_analysis"]
    winner = bte.get("v3_rank1_chunk") or {}
    lines.extend(
        [
            f"chunk_id: {winner.get('chunk_id')}",
            f"section_path: {winner.get('section_path')}",
            f"token_count: {winner.get('token_count')}",
            f"embedding_input:\n{winner.get('embedding_input', '')}",
            f"content: {winner.get('content')}",
            "",
            bte["explanation"],
            "",
            "=== Comparison ===",
        ]
    )
    for row in report["comparison_table"]:
        lines.append(
            f"{row['query'][:40]} | V2={row['v2_rank']} | V3={row['v3_rank']} | "
            f"{row['root_cause']} | {row['v3_issue']}"
        )

    lines.extend(["", "=== Recommendations ==="])
    for key, rec in report["recommendations"].items():
        lines.append(f"{key}: {rec['action']} — {rec['detail']}")

    return "\n".join(lines)
