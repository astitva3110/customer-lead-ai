from __future__ import annotations

from app.services.retrieval.models import HybridSearchResult, RetrievalCandidate


def _line(index: int, candidate: RetrievalCandidate, score_name: str, score: float | None) -> str:
    rendered = "None" if score is None else f"{score:.4f}"
    page = candidate.page_number
    page_bit = f" page={page}" if page is not None else ""
    return f"{index}. {candidate.chunk_id} {score_name}={rendered}{page_bit}"


def format_retrieval_trace(result: HybridSearchResult) -> str:
    """Debug trace with IDs and scores only — not full chunk text."""
    lines = [
        "QUERY:",
        f'"{result.query}"',
        "",
        "VECTOR:",
    ]
    if result.vector_candidates:
        lines.extend(
            _line(index, candidate, "score", candidate.vector_score)
            for index, candidate in enumerate(result.vector_candidates, start=1)
        )
    else:
        lines.append("(none)")

    lines.extend(["", "KEYWORD:"])
    if result.keyword_candidates:
        lines.extend(
            _line(index, candidate, "score", candidate.keyword_score)
            for index, candidate in enumerate(result.keyword_candidates, start=1)
        )
    else:
        lines.append("(none)")

    lines.extend(["", "MERGED:"])
    if result.merged_candidates:
        for index, candidate in enumerate(result.merged_candidates, start=1):
            lines.append(
                f"{index}. {candidate.chunk_id} "
                f"vector={candidate.vector_score} keyword={candidate.keyword_score}"
            )
    else:
        lines.append("(none)")

    lines.extend(["", "RERANKED:"])
    if result.reranked_candidates:
        lines.extend(
            _line(index, candidate, "rerank", candidate.rerank_score)
            for index, candidate in enumerate(result.reranked_candidates, start=1)
        )
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "THRESHOLD:",
            str(result.min_score),
            "",
            "FINAL:",
        ]
    )
    if result.final_candidates:
        lines.extend(candidate.chunk_id for candidate in result.final_candidates)
    else:
        lines.append("(none)")
    lines.extend(
        [
            "",
            f"has_relevant_context={result.has_relevant_context}",
            f"timings={result.timings.to_dict()}",
        ]
    )
    return "\n".join(lines)
