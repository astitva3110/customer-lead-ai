"""Text report formatting for retrieval root-cause diagnostics."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.retrieval_diagnostic.ranking import top_competing_chunks


def format_text_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Earkart Retrieval Root-Cause Diagnostic (Phase 11.7)",
        f"Generated at: {report['generated_at']}",
        f"Dataset: {report['evaluation']['dataset']}",
        f"Index chunks: {report['evaluation']['index']['chunk_count']}",
        "",
        "Executive summary",
        "-----------------",
        f"Questions: {report['evaluation']['question_count']}",
        f"Expected chunk in top-10: {summary['expected_chunks_found_in_top_10']}/{report['evaluation']['question_count']}",
        f"Expected chunk in top-100: {summary['expected_chunks_found_in_top_100']}/{report['evaluation']['question_count']}",
        f"Expected document in top-10: {summary['expected_documents_found_in_top_10']}/{report['evaluation']['question_count']}",
        f"Expected document in top-100: {summary['expected_documents_found_in_top_100']}/{report['evaluation']['question_count']}",
        "",
        "Root-cause counts",
        "-----------------",
    ]
    for label, count in summary["root_cause_counts"].items():
        lines.append(f"{label}: {count}")

    lines.extend(["", "Per-question diagnosis", "--------------------"])
    for question in report["questions"]:
        diagnosis = question["diagnosis"]
        chunk = question["expected_chunk_analysis"]
        document = question["expected_document_analysis"]
        keyword = question["keyword_analysis"]
        whole = question["whole_document_analysis"]
        lines.extend(
            [
                "",
                f"{question['id']}: {question['question']}",
                f"  Primary root cause: {diagnosis['primary_root_cause']} (confidence={diagnosis['confidence']})",
                f"  Secondary: {', '.join(diagnosis['secondary_root_causes']) or 'none'}",
                f"  Expected chunk rank: {chunk.get('best_expected_chunk_rank') or 'not in top-100'}",
                f"  Expected document rank: {document.get('expected_document_best_rank') or 'not in top-100'}",
                f"  BM25 expected chunk rank: {keyword.get('expected_chunk_best_rank') or 'not in top-100'}",
                f"  Whole-document similarity: {whole.get('similarity')}",
                f"  Whole-document rank: {whole.get('rank_among_documents')}",
            ]
        )
        if chunk.get("best_expected_chunk_rank") is None or chunk.get("best_expected_chunk_rank") > 10:
            lines.append("  Why not in top-10:")
            for item in diagnosis["evidence"]:
                lines.append(f"    - {item}")
        competing = top_competing_chunks(
            question["vector_search"]["top_100"],
            expected_chunk_ids=set(question["expected"]["expected_chunk_ids"]),
            expected_document_ids=set(question["expected"]["expected_document_ids"]),
        )
        if competing:
            lines.append("  Top competing chunks:")
            for item in competing:
                lines.append(
                    f"    #{item['rank']} sim={item.get('similarity')} doc={item.get('document_id')} title={item.get('title')!r}"
                )
        lines.append(f"  Recommendation: {diagnosis['recommendation']}")

    lines.extend(
        [
            "",
            "EAR_KART_RETRIEVAL_ROOT_CAUSE_13",
            "",
            f"Questions: {report['evaluation']['question_count']}",
            "",
            f"Expected chunk in Top-10: {summary['expected_chunks_found_in_top_10']}/{report['evaluation']['question_count']}",
            f"Expected chunk in Top-20: {summary['expected_chunks_found_in_top_20']}/{report['evaluation']['question_count']}",
            f"Expected chunk in Top-50: {summary['expected_chunks_found_in_top_50']}/{report['evaluation']['question_count']}",
            f"Expected chunk in Top-100: {summary['expected_chunks_found_in_top_100']}/{report['evaluation']['question_count']}",
            "",
            f"Expected document in Top-10: {summary['expected_documents_found_in_top_10']}/{report['evaluation']['question_count']}",
            f"Expected document in Top-20: {summary['expected_documents_found_in_top_20']}/{report['evaluation']['question_count']}",
            f"Expected document in Top-50: {summary['expected_documents_found_in_top_50']}/{report['evaluation']['question_count']}",
            f"Expected document in Top-100: {summary['expected_documents_found_in_top_100']}/{report['evaluation']['question_count']}",
            "",
            "Root causes:",
        ]
    )
    for label, count in summary["root_cause_counts"].items():
        lines.append(f"{label}: {count}")
    lines.append("")
    lines.append(f"Final verdict: {report['final_verdict']}")
    return "\n".join(lines)
