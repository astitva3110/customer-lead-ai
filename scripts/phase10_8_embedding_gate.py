"""Phase 10.8 — Final chunk embedding gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.kb.chunking.config import ChunkingConfig
from app.kb.chunking.gate import EmbeddingGateService
from app.kb.chunking.storage import ChunkStore, chunk_manifest_path
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.retrieval.storage import RetrievalStore


def _severity_counts(result) -> dict[str, int]:
    counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    real_p0 = [f for f in result.p0_findings if f.get("classification", "").startswith("REAL_")]
    counts["P0"] = len(real_p0)
    counts["P1"] = len(result.p1_findings)
    counts["P2"] = len(result.p2_findings)
    return counts


def main() -> None:
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    retrieval_store = RetrievalStore(settings.retrieval_dir)
    chunk_store = ChunkStore(settings.chunks_dir)
    config = ChunkingConfig(
        max_chunk_tokens=settings.chunk_max_tokens,
        emergency_overlap_tokens=settings.chunk_emergency_overlap_tokens,
        chars_per_token=settings.chunk_chars_per_token,
    )
    gate_service = EmbeddingGateService(
        retrieval_store,
        chunk_store,
        kb_dataset_version=KB_DATASET_VERSION,
        config=config,
    )
    result = gate_service.run(write_production=True)

    severity = _severity_counts(result)
    manifest_path = chunk_manifest_path(settings.chunks_dir, KB_DATASET_VERSION)

    gate_report = {
        "phase": "10.8",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kb_dataset_version": KB_DATASET_VERSION,
        "ready_for_embeddings": result.ready_for_embeddings,
        "gate_passed": result.gate_passed,
        "severity": severity,
        "statistics": result.statistics,
        "validation": result.validation,
        "p0_findings": result.p0_findings,
        "p1_findings": result.p1_findings,
        "p2_findings": result.p2_findings,
        "question_coverage": {
            "total": len(result.question_coverage),
            "passed": sum(1 for q in result.question_coverage if q["passed"]),
            "results": result.question_coverage,
        },
        "production_manifest_created": manifest_path.exists() and result.gate_passed,
        "production_manifest_path": str(manifest_path) if manifest_path.exists() else None,
    }

    (reports_dir / "phase10_8_embedding_gate.json").write_text(
        json.dumps(gate_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "phase10_8_excluded_chunks.json").write_text(
        json.dumps(
            {
                "generated_at": gate_report["generated_at"],
                "total": len(result.excluded_records),
                "chunks": result.excluded_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports_dir / "phase10_8_review_chunks.json").write_text(
        json.dumps(
            {
                "generated_at": gate_report["generated_at"],
                "total": len(result.review_records),
                "chunks": result.review_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports_dir / "phase10_8_question_coverage.json").write_text(
        json.dumps(gate_report["question_coverage"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = result.statistics
    lines = [
        "Phase 10.8 — Final Chunk Embedding Gate",
        f"Generated: {gate_report['generated_at']}",
        f"KB dataset version: {KB_DATASET_VERSION}",
        "",
        "Candidate chunks:",
        f"  Total: {stats['candidate_total']}",
        f"  EMBED_READY: {stats['candidate_by_status'].get('EMBED_READY', 0)}",
        f"  REVIEW: {stats['candidate_by_status'].get('REVIEW', 0)}",
        f"  DO_NOT_EMBED: {stats['candidate_by_status'].get('DO_NOT_EMBED', 0)}",
        "",
        "Production embedding manifest:",
        f"  Total EMBED_READY: {stats['production_total']}",
        f"  Production percentage: {stats['production_percentage']}%",
        f"  Review (audit only): {stats['review_total']}",
        f"  Excluded (audit only): {stats['excluded_total']}",
        f"  Engine pre-filtered: {stats['engine_suppressed_total']}",
        "",
        f"P0 findings: {severity['P0']}",
        f"P1 findings: {severity['P1']}",
        f"P2 findings: {severity['P2']}",
        "",
        "Source fidelity (P0 investigation):",
    ]
    for finding in result.p0_findings:
        lines.append(
            f"  {finding.get('document_id', '?')} / {finding.get('chunk_id', '?')}: "
            f"{finding.get('classification')} — {finding.get('detail')}"
        )
    lines.extend(
        [
            "",
            f"Question coverage: {gate_report['question_coverage']['passed']}/{gate_report['question_coverage']['total']}",
            f"Production manifest created: {gate_report['production_manifest_created']}",
            "",
            f"READY_FOR_EMBEDDINGS: {'YES' if result.ready_for_embeddings else 'NO'}",
        ]
    )
    (reports_dir / "phase10_8_embedding_gate.txt").write_text("\n".join(lines), encoding="utf-8")

    if not result.gate_passed and manifest_path.exists():
        manifest_path.unlink()

    print("\n".join(lines))


if __name__ == "__main__":
    main()
