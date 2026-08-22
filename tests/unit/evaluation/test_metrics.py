from __future__ import annotations

from app.evaluation.metrics import aggregate_retrieval, latency_percentiles, layer_metrics, percentile, retrieval_turn_metrics
from app.evaluation.classify import TurnClassification


def test_aggregation_and_latency():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    summary = latency_percentiles(values)
    assert summary["p50"] == 30.0
    assert summary["p99"] == 50.0
    retrieval = aggregate_retrieval(
        [
            {"recall_at_1": 1, "recall_at_3": 1, "recall_at_5": 1, "recall_at_10": 1, "mrr": 1, "passed": True, "bucket": "Product"},
            {"recall_at_1": 0, "recall_at_3": 1, "recall_at_5": 1, "recall_at_10": 1, "mrr": 0.5, "passed": True, "bucket": "Follow-up"},
        ]
    )
    assert retrieval["recall_at_10"] == 1.0
    assert "Product" in retrieval["by_category"]
    layers = layer_metrics(
        [
            TurnClassification("GEN-1", 1, "KNOWLEDGE_DIRECT", "NONE", None, True, "OK"),
            TurnClassification("GEN-2", 1, "KNOWLEDGE_DIRECT", "RETRIEVAL_FAILURE", "retrieval", False, "EMPTY_RETRIEVAL"),
        ]
    )
    assert layers["retrieval"]["failures"] == 1
    assert layers["guardrail"]["failures"] == 0
    assert "pass_rate" not in layers["retrieval"]


def test_retrieval_metrics_require_verified_catalog_key_and_retrieval(catalog):
    from app.evaluation.catalog import items_by_key
    from app.evaluation.schema import TurnMessage
    from tests.unit.evaluation.helpers import knowledge_trace

    mapping = items_by_key(catalog)
    message = TurnMessage(turn=1, text="What is Signia?", needs_rag=True, knowledge_keys=["product:signia"])
    assert retrieval_turn_metrics(message, knowledge_trace(), mapping) is None
    verified = TurnMessage(turn=1, text="What is TINY?", needs_rag=True, knowledge_keys=["product:tiny"])
    empty = knowledge_trace(raw=[], final=[])
    assert retrieval_turn_metrics(verified, empty, mapping) is None
    scored = retrieval_turn_metrics(verified, knowledge_trace(), mapping)
    assert scored is not None
    assert scored["ground_truth_source"] == "verified_knowledge_key"
