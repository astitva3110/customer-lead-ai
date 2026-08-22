"""Write V3.2 corpus integrity reports."""

from __future__ import annotations

import json
from pathlib import Path

from app.kb.ingestion.phase16_corpus_v3_2 import V32IntegrityReport


def format_integrity_report(report: V32IntegrityReport) -> str:
    payload = report.to_dict()
    lines = [
        "V3.2 INTEGRITY REPORT",
        f"Passed: {payload['passed']}",
        "",
        f"V3.1 total chunks: {payload['v3_1_total_chunks']}",
        f"V3.1 atomic chunks: {payload['v3_1_atomic_chunks']}",
        f"V3.1 summary chunks: {payload['v3_1_summary_chunks']}",
        f"V3.2 total chunks: {payload['v3_2_total_chunks']}",
        f"V3.2 new summary chunks: {payload['v3_2_new_summary_chunks']}",
        f"Unchanged V3.1 chunks: {payload['unchanged_v3_1_chunks']}",
        f"Changed existing chunks: {payload['changed_existing_chunks']}",
    ]
    if payload["details"]:
        lines.extend(["", "Details:"])
        for detail in payload["details"]:
            lines.append(f"  - {detail}")
    return "\n".join(lines)


def write_integrity_report(*, reports_dir: Path, report: V32IntegrityReport) -> tuple[Path, Path]:
    out_dir = reports_dir / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "earkart_v3_2_integrity.json"
    txt_path = out_dir / "earkart_v3_2_integrity.txt"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    txt_path.write_text(format_integrity_report(report), encoding="utf-8")
    return json_path, txt_path
