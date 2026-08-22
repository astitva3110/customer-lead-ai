from __future__ import annotations

from app.evaluation.classify import TurnClassification
from app.evaluation.cluster import cluster_failures, select_representatives


def _fail(cid: str, category: str, product: str, trace: str, pattern="STATE_FAILURE_PRODUCT_SWITCH") -> TurnClassification:
    return TurnClassification(
        conversation_id=cid,
        turn=3,
        category=category,
        failure_type="STATE_FAILURE",
        first_failing_layer="state",
        passed=False,
        pattern=pattern,
        product=product,
        knowledge_keys=["product:tiny"],
        trace_id=trace,
        details={"from": "TINY", "to": "Radius M16"},
    )


def test_failure_clustering():
    rows = [
        _fail("GEN-000123", "LEAD_PRODUCT_SWITCH", "Bluup", "tr-a"),
        _fail("GEN-000287", "LEAD_PRODUCT_SWITCH", "Bluup", "tr-b"),
        _fail("GEN-000441", "LEAD_PRODUCT_SWITCH", "Radius M16", "tr-c"),
        TurnClassification(
            conversation_id="GEN-000009",
            turn=1,
            category="KNOWLEDGE_DIRECT",
            failure_type="RETRIEVAL_FAILURE",
            first_failing_layer="retrieval",
            passed=False,
            pattern="EXPECTED_NOT_IN_RAW",
            trace_id="tr-d",
        ),
    ]
    clusters = cluster_failures(rows, total_conversations=100)
    assert clusters[0]["count"] == 3
    assert clusters[0]["percentage"] == 3.0
    assert clusters[0]["common_message_pattern"] == "STATE_FAILURE_PRODUCT_SWITCH"
    assert clusters[0]["common_transition"] == "TINY -> Radius M16"
    assert clusters[0]["first_failing_layer"] == "state"
    assert "GEN-000123" in clusters[0]["representative_conversation_ids"]


def test_representative_trace_selection_caps():
    rows = [_fail(f"GEN-{index:06d}", "LEAD_PRODUCT_SWITCH", "TINY" if index % 2 else "Bluup", f"tr-{index}") for index in range(1, 20)]
    selected = select_representatives(rows, limit=5)
    assert 3 <= len(selected) <= 5
    assert len({item.conversation_id for item in selected}) == len(selected)
