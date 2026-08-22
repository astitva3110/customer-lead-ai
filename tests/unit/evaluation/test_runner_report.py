from __future__ import annotations

import json

from app.evaluation.client import TurnOutcome
from app.evaluation.generator import generate_dataset
from app.evaluation.report import format_run_text, write_mass_eval_reports
from app.evaluation.runner import aggregate_run, evaluate_conversation, run_mass_evaluation
from tests.unit.evaluation.helpers import make_conversation, passing_handle


def test_clusters_use_first_failure_not_last_turn(eval_config):
    from app.evaluation.schema import GeneratedDataset

    conversation = make_conversation(conversation_id="GEN-000001")
    dataset = GeneratedDataset(
        dataset_version="mass-test",
        evaluator_version="1.0.0",
        seed=42,
        count=1,
        generator_model="template",
        corpus_version="v3.2",
        knowledge_catalog="data/evaluations/earkart_kb_v3_1_comprehensive.json",
        conversations=[conversation],
    )
    results = [
        {
            "conversation_id": "GEN-000001",
            "category": "KNOWLEDGE_FOLLOWUP",
            "passed": False,
            "failure_type": "RETRIEVAL_FAILURE",
            "first_failing_layer": "retrieval",
            "pattern": "EXPECTED_NOT_IN_RAW",
            "trace_ids": ["tr-1", "tr-2"],
            "goal": "KNOWLEDGE",
            "turns": [
                {
                    "turn": 1,
                    "trace_id": "tr-1",
                    "classification": {
                        "turn": 1,
                        "failure_type": "RETRIEVAL_FAILURE",
                        "first_failing_layer": "retrieval",
                        "passed": False,
                        "pattern": "EXPECTED_NOT_IN_RAW",
                        "product": "TINY",
                        "knowledge_keys": ["product:tiny"],
                        "trace_id": "tr-1",
                        "details": {},
                    },
                },
                {
                    "turn": 2,
                    "trace_id": "tr-2",
                    "classification": {
                        "turn": 2,
                        "failure_type": "NONE",
                        "first_failing_layer": None,
                        "passed": True,
                        "pattern": "OK",
                        "product": "TINY",
                        "knowledge_keys": [],
                        "trace_id": "tr-2",
                        "details": {},
                    },
                },
            ],
        }
    ]
    summary = aggregate_run(dataset, results, eval_config, "cluster-test")
    assert summary["failure_clusters"][0]["failure_type"] == "RETRIEVAL_FAILURE"
    assert summary["failure_clusters"][0]["representative_trace_ids"] == ["tr-1"]



def test_report_generation(tmp_path):
    summary = {
        "run_id": "test-run",
        "dataset_version": "mass-test",
        "conversation_count": 2,
        "turn_count": 6,
        "conversations_with_candidate_failures": 1,
        "layers": {"retrieval": {"failures": 1, "candidate_failures": 1}},
        "failure_clusters": [
            {
                "cluster_id": "RETRIEVAL_FAILURE::KNOWLEDGE_DIRECT::EMPTY_RETRIEVAL",
                "failure_type": "RETRIEVAL_FAILURE",
                "count": 1,
                "percentage": 50.0,
                "first_failing_layer": "retrieval",
                "common_message_pattern": "EMPTY_RETRIEVAL",
                "representative_conversation_ids": ["GEN-000001"],
                "representative_trace_ids": ["tr-1"],
            }
        ],
        "category_metrics": {},
        "representative_failures": [],
        "retrieval": {"recall_at_5": 0.4, "recall_at_10": 0.5, "mrr": 0.3},
        "latency": {"all": {"p50": 1, "p90": 2, "p95": 3, "p99": 4}},
        "llm_calls": {"total": 9},
    }
    reports = write_mass_eval_reports(tmp_path, "test-run", summary, [{"conversation_id": "GEN-000001", "candidate": True, "has_verified_candidate_failure": True}])
    assert reports["summary_json"].exists()
    payload = json.loads(reports["summary_json"].read_text(encoding="utf-8"))
    assert payload["run_id"] == "test-run"
    assert "overall_pass_rate" not in payload
    assert payload["failure_clusters"]
    text = format_run_text(payload)
    assert "CANDIDATE FAILURE CLUSTERS" in text
    assert "overall_pass_rate" not in text
    assert reports["run_txt"].exists()
    assert reports["candidates_json"].exists()


def test_resume_behavior(eval_config, tmp_path):
    dataset = generate_dataset(eval_config, count=4, use_llm=False)
    seen: list[str] = []

    def flaky(conversation_id: str, message: str) -> TurnOutcome:
        if conversation_id == "GEN-000003" and conversation_id not in seen:
            seen.append(conversation_id)
            raise RuntimeError("boom")
        return passing_handle(conversation_id, message)

    eval_config_resume = eval_config
    first = run_mass_evaluation(dataset, eval_config_resume, handle=flaky, run_id="resume-test", force=True)
    assert "GEN-000003" in first["completed_ids"]
    assert "overall_pass_rate" not in first
    calls_before = {"n": 0}

    def counting(conversation_id: str, message: str) -> TurnOutcome:
        calls_before["n"] += 1
        return passing_handle(conversation_id, message)

    second = run_mass_evaluation(dataset, eval_config_resume, handle=counting, run_id="resume-test", resume=True)
    assert second["conversation_count"] == 4
    assert calls_before["n"] == 0


def test_evaluate_conversation_associates_trace(eval_config, catalog):
    dataset = generate_dataset(eval_config, count=1, use_llm=False)
    from app.evaluation.catalog import items_by_key

    result = evaluate_conversation(dataset.conversations[0], passing_handle, eval_config, items_by_key(catalog))
    assert result["trace_ids"]
    assert result["turns"][0]["trace_id"] == "tr-test"
    assert result["turns"][0]["classification"]["failure_type"]
    assert "candidate_failures" in result
    assert "observations" in result
    assert "overall_pass_rate" not in result
