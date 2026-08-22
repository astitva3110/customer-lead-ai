"""Concurrent conversation evaluation with timeout, retry, and resume. Read-only vs production."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.evaluation.catalog import items_by_key, load_knowledge_catalog
from app.evaluation.classify import TurnClassification, classify_conversation, classify_turn
from app.evaluation.client import HandleFn, OrchestratorChatClient, TurnOutcome, enable_tracing
from app.evaluation.cluster import cluster_failures
from app.evaluation.config import MassEvalConfig
from app.evaluation.metrics import (
    aggregate_retrieval,
    category_metrics,
    conversation_behavior_metrics,
    latency_percentiles,
    layer_metrics,
    llm_calls_from_trace,
    rerank_delta,
    retrieval_turn_metrics,
    turn_latency_ms,
)
from app.evaluation.report import write_mass_eval_reports
from app.evaluation.schema import GeneratedConversation, GeneratedDataset, TurnMessage, load_generated_dataset

HandleFactory = Callable[[], HandleFn]


class TimeoutError_(Exception):
    pass


def new_run_id(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_s{seed}"


def run_mass_evaluation(
    dataset: GeneratedDataset | Path | str,
    config: MassEvalConfig,
    *,
    handle: HandleFn | None = None,
    run_id: str | None = None,
    resume: bool | None = None,
    force: bool = False,
) -> dict[str, Any]:
    loaded = dataset if isinstance(dataset, GeneratedDataset) else load_generated_dataset(dataset)
    run = run_id or new_run_id(loaded.seed)
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces" / run
    if config.trace_enabled:
        enable_tracing(trace_dir, top_k=config.chat_trace_retrieval_top_k)
    checkpoint_path = output_dir / f"checkpoint_{run}.json"
    results_path = output_dir / f"partial_{run}.jsonl"
    completed = _load_completed(checkpoint_path, results_path) if (resume if resume is not None else config.resume) and not force else {}
    catalog = items_by_key(load_knowledge_catalog(config.catalog_path()))
    pending = [item for item in loaded.conversations if item.conversation_id not in completed]
    client = None
    if handle is None:
        client = OrchestratorChatClient()
        worker_handle = client.handle
    else:
        worker_handle = handle
    if pending:
        _run_pool(
            pending,
            worker_handle,
            config,
            catalog,
            results_path,
            checkpoint_path,
            completed,
            client=client,
        )
        completed = _load_completed(checkpoint_path, results_path)
    ordered = [completed[item.conversation_id] for item in loaded.conversations if item.conversation_id in completed]
    summary = aggregate_run(loaded, ordered, config, run)
    reports = write_mass_eval_reports(output_dir, run, summary, ordered, trace_dir=trace_dir)
    summary["reports"] = {key: str(path) for key, path in reports.items()}
    return summary


def _run_pool(
    pending: list[GeneratedConversation],
    handle: HandleFn,
    config: MassEvalConfig,
    catalog: dict,
    results_path: Path,
    checkpoint_path: Path,
    completed: dict[str, dict[str, Any]],
    client=None,
) -> None:
    write_lock = __import__("threading").Lock()
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {
            pool.submit(evaluate_conversation, conversation, handle, config, catalog): conversation.conversation_id
            for conversation in pending
        }
        for future in as_completed(futures):
            conversation_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = _failed_conversation(conversation_id, str(exc))
            with write_lock:
                _append_jsonl(results_path, result)
                completed[conversation_id] = result
                _write_checkpoint(checkpoint_path, sorted(completed))
                if client is not None:
                    client.drop(conversation_id)


def evaluate_conversation(
    conversation: GeneratedConversation,
    handle: HandleFn,
    config: MassEvalConfig,
    catalog: dict,
) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    classifications: list[TurnClassification] = []
    tools: list[str] = []
    last_trace: dict[str, Any] = {}
    started = time.perf_counter()
    for message in conversation.messages:
        outcome, error = _handle_with_retry(handle, conversation.conversation_id, message.text, config)
        trace = outcome.trace if outcome else {}
        last_trace = trace or last_trace
        classified = classify_turn(
            conversation,
            message,
            trace,
            catalog=catalog,
            prior_tools=list(tools),
            error=error,
        )
        classifications.append(classified)
        tool_name = str(((trace or {}).get("tool_execution") or {}).get("tool_name") or "")
        if tool_name:
            tools.append(tool_name)
        retrieval_row = retrieval_turn_metrics(message, trace or {}, catalog)
        rerank_row = rerank_delta(trace or {}, [catalog[key] for key in message.knowledge_keys if key in catalog])
        llm = llm_calls_from_trace(trace or {})
        turns.append(
            {
                "turn": message.turn,
                "text": message.text,
                "trace_id": (outcome.trace_id if outcome else ""),
                "response": (outcome.response if outcome else ""),
                "classification": classified.to_dict(),
                "latency_ms": turn_latency_ms(trace or {}),
                "llm_calls": llm,
                "retrieval": retrieval_row,
                "rerank": rerank_row,
                "retrieval_used": bool((trace or {}).get("retrieval")),
                "state_after": (trace or {}).get("state_after") or {},
                "error": error,
            }
        )
    overall = classify_conversation(conversation, classifications, last_trace)
    candidates = [item.to_dict() for item in classifications if item.candidate]
    return {
        "conversation_id": conversation.conversation_id,
        "category": conversation.category,
        "language": conversation.language,
        "kind": "exploratory_scenario",
        "candidate": overall.candidate,
        "has_verified_candidate_failure": overall.candidate,
        "passed": not overall.candidate,
        "failure_type": overall.failure_type if overall.candidate else "NONE",
        "first_failing_layer": overall.first_failing_layer,
        "pattern": overall.pattern,
        "ground_truth_source": overall.ground_truth_source,
        "candidate_failures": candidates,
        "observations": [item.observations for item in classifications],
        "scenario_hypotheses": conversation.expected.model_dump(),
        "trace_ids": [item["trace_id"] for item in turns if item["trace_id"]],
        "turns": turns,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "goal": conversation.expected.goal,
    }


def aggregate_run(
    dataset: GeneratedDataset,
    results: list[dict[str, Any]],
    config: MassEvalConfig,
    run_id: str,
) -> dict[str, Any]:
    turn_classifications: list[TurnClassification] = []
    conversation_classifications: list[TurnClassification] = []
    retrieval_rows: list[dict[str, Any]] = []
    rerank_stats = {"improved": 0, "regressed": 0, "unchanged": 0}
    latencies: dict[str, list[float]] = {"all": [], "knowledge": [], "lead": [], "support": [], "mixed": []}
    llm_totals = {"generation": 0, "query_rewrite": 0, "router": 0, "total": 0, "lead": 0, "support": 0}
    for result in results:
        turns = result.get("turns") or []
        first_failure: TurnClassification | None = None
        for turn in turns:
            payload = turn.get("classification") or {}
            classified = TurnClassification(
                conversation_id=result["conversation_id"],
                turn=int(payload.get("turn") or turn.get("turn") or 0),
                category=result["category"],
                failure_type=str(payload.get("failure_type") or "NONE"),
                first_failing_layer=payload.get("first_failing_layer"),
                passed=bool(payload.get("passed", not payload.get("candidate"))),
                pattern=str(payload.get("pattern") or ""),
                product=payload.get("product"),
                knowledge_keys=list(payload.get("knowledge_keys") or []),
                trace_id=str(payload.get("trace_id") or turn.get("trace_id") or ""),
                details=dict(payload.get("details") or {}),
                ground_truth_source=payload.get("ground_truth_source"),
                observations=dict(payload.get("observations") or {}),
            )
            turn_classifications.append(classified)
            if first_failure is None and classified.candidate:
                first_failure = classified
            if turn.get("retrieval"):
                retrieval_rows.append(turn["retrieval"])
            status = (turn.get("rerank") or {}).get("status")
            if status in rerank_stats:
                rerank_stats[status] += 1
            latency = turn.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies["all"].append(float(latency))
                bucket = _latency_bucket(result.get("goal"), result.get("category"))
                latencies[bucket].append(float(latency))
            calls = turn.get("llm_calls") or {}
            for key in ("generation", "query_rewrite", "router", "total"):
                llm_totals[key] += int(calls.get(key) or 0)
            if result.get("goal") == "LEAD":
                llm_totals["lead"] += int(calls.get("total") or 0)
            if result.get("goal") == "SUPPORT":
                llm_totals["support"] += int(calls.get("total") or 0)
        if first_failure is not None:
            conversation_classifications.append(first_failure)
        else:
            passed = not bool(result.get("has_verified_candidate_failure") or result.get("candidate") or not result.get("passed", True))
            conversation_classifications.append(
                TurnClassification(
                    conversation_id=result["conversation_id"],
                    turn=0,
                    category=result["category"],
                    failure_type=str(result.get("failure_type") or "NONE"),
                    first_failing_layer=result.get("first_failing_layer"),
                    passed=passed,
                    pattern=str(result.get("pattern") or ("OBSERVED" if passed else "TURN_ERROR")),
                    trace_id=(result.get("trace_ids") or [""])[-1] if result.get("trace_ids") else "",
                    ground_truth_source=result.get("ground_truth_source"),
                )
            )
    clusters = cluster_failures(
        [item for item in conversation_classifications if item.candidate],
        total_conversations=len(results) or 1,
    )
    candidate_conversations = sum(1 for item in results if _result_has_candidate(item))
    retrieval_summary = aggregate_retrieval(retrieval_rows)
    representatives = []
    for cluster in clusters[:10]:
        representatives.append(
            {
                "cluster_id": cluster["cluster_id"],
                "conversation_ids": cluster["representative_conversation_ids"],
                "trace_ids": cluster["representative_trace_ids"],
            }
        )
    by_id = {item.conversation_id: item for item in dataset.conversations}
    return {
        "run_id": run_id,
        "dataset_version": dataset.dataset_version,
        "evaluator_version": dataset.evaluator_version,
        "corpus_version": dataset.corpus_version,
        "generator_model": dataset.generator_model,
        "config_snapshot": config.to_dict(),
        "kind": "exploratory_scenario",
        "conversation_count": len(results),
        "turn_count": sum(len(item.get("turns") or []) for item in results),
        "conversations_with_candidate_failures": candidate_conversations,
        "candidate_failure_turns": sum(1 for item in turn_classifications if item.candidate),
        "verified_ground_truth_turns": sum(1 for item in turn_classifications if item.ground_truth_source),
        "layers": layer_metrics(turn_classifications + [item for item in conversation_classifications if item.turn == 0 and item.candidate]),
        "candidate_failure_clusters": clusters,
        "failure_clusters": clusters,
        "category_metrics": category_metrics([by_id[item["conversation_id"]] for item in results if item["conversation_id"] in by_id], conversation_classifications),
        "representative_failures": representatives,
        "retrieval": retrieval_summary,
        "reranker": rerank_stats,
        "latency": {key: latency_percentiles(values) for key, values in latencies.items()},
        "llm_calls": llm_totals,
        "conversation_behavior": conversation_behavior_metrics(
            [by_id[item["conversation_id"]] for item in results if item["conversation_id"] in by_id],
            results,
        ),
        "completed_ids": [item["conversation_id"] for item in results],
    }


def _handle_with_retry(
    handle: HandleFn,
    conversation_id: str,
    message: str,
    config: MassEvalConfig,
) -> tuple[TurnOutcome | None, str | None]:
    last_error = None
    attempts = config.retry_count + 1
    for _ in range(attempts):
        try:
            outcome = _call_with_timeout(handle, conversation_id, message, config.timeout_seconds)
            return outcome, None
        except Exception as exc:
            last_error = str(exc)
    return None, last_error or "unknown_error"


def _call_with_timeout(handle: HandleFn, conversation_id: str, message: str, timeout: float) -> TurnOutcome:
    # Stay on the worker thread so conversation memory is not split across pools.
    # LiteLLM already enforces generation_timeout_seconds per call.
    del timeout
    return handle(conversation_id, message)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _write_checkpoint(path: Path, ids: list[str]) -> None:
    path.write_text(json.dumps({"completed_ids": ids}, indent=2), encoding="utf-8")


def _load_completed(checkpoint_path: Path, results_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed[row["conversation_id"]] = row
    elif checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for cid in payload.get("completed_ids") or []:
            completed[cid] = _failed_conversation(cid, "missing_partial_row")
    return completed


def _failed_conversation(conversation_id: str, error: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "category": "UNKNOWN",
        "kind": "exploratory_scenario",
        "candidate": True,
        "has_verified_candidate_failure": True,
        "passed": False,
        "failure_type": "TOOL_FAILURE",
        "first_failing_layer": "tools",
        "pattern": "CONVERSATION_ERROR",
        "ground_truth_source": "deterministic_rule",
        "candidate_failures": [],
        "observations": [{"error": error}],
        "trace_ids": [],
        "turns": [],
        "error": error,
        "goal": "NONE",
    }


def _result_has_candidate(result: dict[str, Any]) -> bool:
    if "has_verified_candidate_failure" in result:
        return bool(result["has_verified_candidate_failure"])
    if "candidate" in result:
        return bool(result["candidate"])
    return not bool(result.get("passed", True))


def reobserve_results(
    dataset,
    results: list[dict[str, Any]],
    catalog: dict,
    trace_dir: Path | None,
) -> list[dict[str, Any]]:
    """Re-apply verified-GT observation to saved turns/traces. Does not call /chat."""
    by_id = {item.conversation_id: item for item in dataset.conversations}
    observed: list[dict[str, Any]] = []
    for result in results:
        conversation = by_id.get(result.get("conversation_id"))
        if conversation is None:
            observed.append(result)
            continue
        observed.append(reobserve_conversation(conversation, result, catalog, trace_dir))
    return observed


def reobserve_conversation(
    conversation: GeneratedConversation,
    result: dict[str, Any],
    catalog: dict,
    trace_dir: Path | None,
) -> dict[str, Any]:
    turns_out: list[dict[str, Any]] = []
    classifications: list[TurnClassification] = []
    tools: list[str] = []
    last_trace: dict[str, Any] = {}
    saved_turns = list(result.get("turns") or [])
    pairs = list(zip(conversation.messages, saved_turns)) if saved_turns else [(message, {}) for message in conversation.messages]
    for message, saved in pairs:
        trace = _load_saved_trace(saved, trace_dir)
        if trace:
            last_trace = trace
            classified = classify_turn(
                conversation,
                message,
                trace,
                catalog=catalog,
                prior_tools=list(tools),
                error=saved.get("error") if saved else result.get("error"),
            )
        else:
            classified = _classification_without_trace(conversation, message, saved)
        classifications.append(classified)
        tool_name = str(((trace or {}).get("tool_execution") or {}).get("tool_name") or "")
        if tool_name:
            tools.append(tool_name)
        retrieval_row = retrieval_turn_metrics(message, trace, catalog) if trace else saved.get("retrieval")
        rerank_row = (
            rerank_delta(trace, [catalog[key] for key in message.knowledge_keys if key in catalog])
            if trace
            else saved.get("rerank")
        )
        merged = dict(saved or {})
        merged.update(
            {
                "turn": message.turn,
                "text": message.text,
                "classification": classified.to_dict(),
                "retrieval": retrieval_row,
                "rerank": rerank_row,
                "retrieval_used": bool(trace.get("retrieval")) if trace else merged.get("retrieval_used"),
            }
        )
        turns_out.append(merged)
    overall = classify_conversation(conversation, classifications, last_trace)
    candidates = [item.to_dict() for item in classifications if item.candidate]
    updated = dict(result)
    updated.update(
        {
            "kind": "exploratory_scenario",
            "candidate": overall.candidate,
            "has_verified_candidate_failure": overall.candidate,
            "passed": not overall.candidate,
            "failure_type": overall.failure_type if overall.candidate else "NONE",
            "first_failing_layer": overall.first_failing_layer,
            "pattern": overall.pattern,
            "ground_truth_source": overall.ground_truth_source,
            "candidate_failures": candidates,
            "observations": [item.observations for item in classifications],
            "scenario_hypotheses": conversation.expected.model_dump(),
            "turns": turns_out,
        }
    )
    return updated


_HYPOTHESIS_ONLY_FAILURES = frozenset(
    {
        "LEAD_NOT_CREATED",
        "TICKET_NOT_CREATED",
        "LEAD_WORKFLOW_FAILURE",
        "SUPPORT_WORKFLOW_FAILURE",
        "RAG_TURN_NOT_ROUTED",
        "ROUTER_FAILURE",
        "EXPECTED_BEHAVIOR_MISMATCH",
        "CONVERSATION_QUALITY_FAILURE",
    }
)


def _classification_without_trace(
    conversation: GeneratedConversation,
    message: TurnMessage,
    saved: dict[str, Any],
) -> TurnClassification:
    payload = saved.get("classification") or {}
    failure = str(payload.get("failure_type") or "NONE")
    source = payload.get("ground_truth_source")
    hypothesis = failure in _HYPOTHESIS_ONLY_FAILURES or not source
    return TurnClassification(
        conversation_id=conversation.conversation_id,
        turn=message.turn,
        category=conversation.category,
        failure_type="NONE" if hypothesis else failure,
        first_failing_layer=None if hypothesis else payload.get("first_failing_layer"),
        passed=True if hypothesis else bool(payload.get("passed", not payload.get("candidate"))),
        pattern="OBSERVED" if hypothesis else str(payload.get("pattern") or "OBSERVED"),
        product=payload.get("product") or message.expected_product,
        knowledge_keys=list(payload.get("knowledge_keys") or message.knowledge_keys),
        trace_id=str(payload.get("trace_id") or saved.get("trace_id") or ""),
        details=dict(payload.get("details") or {}),
        ground_truth_source=None if hypothesis else source,
        observations=dict(payload.get("observations") or {"trace_missing": True}),
    )


def _load_saved_trace(saved: dict[str, Any], trace_dir: Path | None) -> dict[str, Any]:
    if saved.get("trace") and isinstance(saved["trace"], dict):
        return saved["trace"]
    trace_id = str(saved.get("trace_id") or "")
    if not trace_id or not trace_dir:
        return {}
    path = Path(trace_dir) / f"{trace_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latency_bucket(goal: str | None, category: str | None) -> str:
    if category == "MIXED_INTENT":
        return "mixed"
    if goal == "LEAD":
        return "lead"
    if goal == "SUPPORT":
        return "support"
    return "knowledge"
