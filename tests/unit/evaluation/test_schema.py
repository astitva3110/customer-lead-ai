from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.evaluation.generator import generate_dataset
from app.evaluation.schema import load_generated_dataset, validate_dataset_payload


def test_dataset_schema_roundtrip(eval_config, tmp_path):
    dataset = generate_dataset(eval_config, count=20, use_llm=False)
    validate_dataset_payload(dataset)
    path = tmp_path / "v1.json"
    from app.evaluation.schema import dump_generated_dataset

    dump_generated_dataset(dataset, path)
    loaded = load_generated_dataset(path)
    assert loaded.count == 20
    assert len(loaded.conversations) == 20
    assert loaded.conversations[0].conversation_id == "GEN-000001"
    assert loaded.conversations[-1].conversation_id == "GEN-000020"
    assert loaded.kind == "exploratory_scenario"
    assert loaded.conversations[0].kind == "exploratory_scenario"


def test_dataset_rejects_bad_turns(eval_config):
    dataset = generate_dataset(eval_config, count=3, use_llm=False)
    payload = dataset.model_dump()
    payload["conversations"][0]["messages"][1]["turn"] = 9
    with pytest.raises(ValidationError):
        validate_dataset_payload(payload)


def test_existing_generated_dataset_still_loads():
    from pathlib import Path

    path = Path("data/evaluations/generated/v1.json")
    if not path.exists():
        return
    loaded = load_generated_dataset(path)
    assert loaded.count == len(loaded.conversations)
    assert loaded.conversations[0].conversation_id.startswith("GEN-")
    assert loaded.kind == "exploratory_scenario"


def test_deterministic_generation_fixture(eval_config):
    first = generate_dataset(eval_config, count=20, use_llm=False)
    second = generate_dataset(eval_config, count=20, use_llm=False)
    assert [item.conversation_id for item in first.conversations] == [item.conversation_id for item in second.conversations]
    assert [item.category for item in first.conversations] == [item.category for item in second.conversations]
    assert [[msg.text for msg in item.messages] for item in first.conversations] == [
        [msg.text for msg in item.messages] for item in second.conversations
    ]
    categories = {item.category for item in first.conversations}
    assert len(categories) >= 10
    assert all(3 <= len(item.messages) <= 8 for item in first.conversations)
    assert len({tuple(msg.text for msg in item.messages) for item in first.conversations}) >= 15
