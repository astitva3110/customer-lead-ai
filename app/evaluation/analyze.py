"""Re-read an existing mass-eval run. Does not regenerate conversations or re-query the chatbot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.report import format_console_summary


def load_run_summary(output_dir: Path, run_id: str) -> dict[str, Any]:
    path = output_dir / f"summary_{run_id}.json"
    if not path.exists():
        matches = sorted(output_dir.glob(f"summary_*{run_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"no summary for run_id={run_id} under {output_dir}")
        path = matches[-1]
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_run(output_dir: Path, run_id: str) -> dict[str, Any]:
    return load_run_summary(output_dir, run_id)


def reaggregate_run(output_dir: Path, run_id: str, dataset, config) -> dict[str, Any]:
    """Rebuild observation/candidate reports from saved traces without calling /chat."""
    from app.evaluation.catalog import items_by_key, load_knowledge_catalog
    from app.evaluation.report import write_mass_eval_reports
    from app.evaluation.runner import aggregate_run, reobserve_results

    path = output_dir / f"run_{run_id}.json"
    results: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
    else:
        partial = output_dir / f"partial_{run_id}.jsonl"
        if not partial.exists():
            raise FileNotFoundError(f"no run or partial file for run_id={run_id} under {output_dir}")
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
    catalog = items_by_key(load_knowledge_catalog(config.catalog_path()))
    trace_dir = output_dir / "traces" / run_id
    observed = reobserve_results(dataset, results, catalog, trace_dir if trace_dir.exists() else None)
    summary = aggregate_run(dataset, observed, config, run_id)
    reports = write_mass_eval_reports(output_dir, run_id, summary, observed, trace_dir=trace_dir)
    summary["reports"] = {key: str(value) for key, value in reports.items()}
    return summary


def print_analysis(summary: dict[str, Any]) -> str:
    return format_console_summary(summary)
