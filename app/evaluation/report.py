"""Write mass-eval reports. Full traces stay on disk; console gets summaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any



def write_mass_eval_reports(
    output_dir: Path,
    run_id: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    trace_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json = output_dir / f"run_{run_id}.json"
    run_txt = output_dir / f"run_{run_id}.txt"
    summary_json = output_dir / f"summary_{run_id}.json"
    candidates_json = output_dir / f"candidates_{run_id}.json"
    failures_json = output_dir / f"failures_{run_id}.json"
    clusters_json = output_dir / f"clusters_{run_id}.json"
    summary_payload = _summary_payload(summary)
    candidates = [
        item
        for item in results
        if item.get("has_verified_candidate_failure") or item.get("candidate")
    ]
    clusters = list(summary.get("candidate_failure_clusters") or summary.get("failure_clusters") or [])
    run_json.write_text(json.dumps({"summary": summary_payload, "results": results}, indent=2, default=str), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    candidates_json.write_text(json.dumps(candidates, indent=2, default=str), encoding="utf-8")
    failures_json.write_text(json.dumps(candidates, indent=2, default=str), encoding="utf-8")
    clusters_json.write_text(json.dumps(clusters, indent=2, default=str), encoding="utf-8")
    run_txt.write_text(format_run_text(summary_payload), encoding="utf-8")
    copied = copy_representative_traces(output_dir, run_id, clusters, trace_dir)
    return {
        "run_json": run_json,
        "run_txt": run_txt,
        "summary_json": summary_json,
        "candidates_json": candidates_json,
        "failures_json": failures_json,
        "clusters_json": clusters_json,
        "representative_dir": copied,
    }


def copy_representative_traces(
    output_dir: Path,
    run_id: str,
    clusters: list[dict[str, Any]],
    trace_dir: Path | None,
) -> Path:
    dest = output_dir / "representative_traces" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if not trace_dir or not trace_dir.exists():
        return dest
    seen: set[str] = set()
    for cluster in clusters:
        for trace_id in cluster.get("representative_trace_ids") or []:
            if not trace_id or trace_id in seen:
                continue
            seen.add(trace_id)
            for suffix in (".json", ".txt"):
                source = trace_dir / f"{trace_id}{suffix}"
                if source.exists():
                    shutil.copy2(source, dest / f"{trace_id}{suffix}")
    return dest


def format_run_text(summary: dict[str, Any]) -> str:
    retrieval = summary.get("retrieval") or {}
    latency = (summary.get("latency") or {}).get("all") or {}
    llm = summary.get("llm_calls") or {}
    layers = summary.get("layers") or {}
    lines = [
        "EXPLORATORY_SCENARIO_OBSERVATION",
        f"run_id: {summary.get('run_id')}",
        f"dataset_version: {summary.get('dataset_version')}",
        f"kind: {summary.get('kind') or 'exploratory_scenario'}",
        f"conversations: {summary.get('conversation_count')}",
        f"turns: {summary.get('turn_count')}",
        f"conversations_with_candidate_failures: {summary.get('conversations_with_candidate_failures')}",
        f"candidate_failure_turns: {summary.get('candidate_failure_turns')}",
        f"verified_ground_truth_turns: {summary.get('verified_ground_truth_turns')}",
        "",
        "LAYERS (verified candidate failures only)",
    ]
    for name in (
        "guardrail",
        "query_rewrite",
        "retrieval",
        "reranker",
        "generation",
        "grounding",
        "state",
        "tools",
    ):
        row = layers.get(name) or {}
        lines.append(f"  {name}: candidate_failures={row.get('candidate_failures', row.get('failures'))}")
    lines.extend(
        [
            "",
            "RETRIEVAL (verified catalog keys, retrieval ran)",
            f"  recall@1: {retrieval.get('recall_at_1')}",
            f"  recall@3: {retrieval.get('recall_at_3')}",
            f"  recall@5: {retrieval.get('recall_at_5')}",
            f"  recall@10: {retrieval.get('recall_at_10')}",
            f"  mrr: {retrieval.get('mrr')}",
            "",
            "LATENCY",
            f"  p50: {latency.get('p50')}",
            f"  p90: {latency.get('p90')}",
            f"  p95: {latency.get('p95')}",
            f"  p99: {latency.get('p99')}",
            "",
            f"LLM_CALLS: {llm.get('total')}",
            "",
            "CANDIDATE FAILURE CLUSTERS",
        ]
    )
    for index, cluster in enumerate(
        (summary.get("candidate_failure_clusters") or summary.get("failure_clusters") or [])[:8],
        start=1,
    ):
        lines.extend(
            [
                f"{index}. {cluster.get('failure_type')} / {cluster.get('common_message_pattern')}",
                f"   count={cluster.get('count')} percentage={cluster.get('percentage')} layer={cluster.get('first_failing_layer')}",
                f"   conversations={cluster.get('representative_conversation_ids')}",
                f"   traces={cluster.get('representative_trace_ids')}",
            ]
        )
    return "\n".join(lines) + "\n"


def format_console_summary(summary: dict[str, Any]) -> str:
    return format_run_text(summary)


def _summary_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary.get("run_id"),
        "dataset_version": summary.get("dataset_version"),
        "evaluator_version": summary.get("evaluator_version"),
        "corpus_version": summary.get("corpus_version"),
        "kind": summary.get("kind") or "exploratory_scenario",
        "conversation_count": summary.get("conversation_count"),
        "turn_count": summary.get("turn_count"),
        "conversations_with_candidate_failures": summary.get("conversations_with_candidate_failures"),
        "candidate_failure_turns": summary.get("candidate_failure_turns"),
        "verified_ground_truth_turns": summary.get("verified_ground_truth_turns"),
        "layers": summary.get("layers") or {},
        "candidate_failure_clusters": summary.get("candidate_failure_clusters") or summary.get("failure_clusters") or [],
        "failure_clusters": summary.get("candidate_failure_clusters") or summary.get("failure_clusters") or [],
        "category_metrics": summary.get("category_metrics") or {},
        "representative_failures": summary.get("representative_failures") or [],
        "retrieval": summary.get("retrieval") or {},
        "reranker": summary.get("reranker") or {},
        "latency": summary.get("latency") or {},
        "llm_calls": summary.get("llm_calls") or {},
        "conversation_behavior": summary.get("conversation_behavior") or {},
        "config_snapshot": summary.get("config_snapshot") or {},
    }
