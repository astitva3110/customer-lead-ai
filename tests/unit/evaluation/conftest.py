from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.catalog import load_knowledge_catalog
from app.evaluation.config import MassEvalConfig


@pytest.fixture
def eval_config(tmp_path: Path) -> MassEvalConfig:
    return MassEvalConfig.from_dict(
        {
            "dataset_version": "mass-test",
            "count": 20,
            "seed": 42,
            "use_llm": False,
            "min_turns": 3,
            "max_turns": 8,
            "concurrency": 2,
            "timeout_seconds": 5,
            "retry_count": 0,
            "resume": True,
            "output_dir": str(tmp_path / "reports"),
            "generated_output": str(tmp_path / "generated.json"),
        }
    )


@pytest.fixture
def catalog():
    return load_knowledge_catalog(Path("data/evaluations/earkart_kb_v3_1_comprehensive.json"))
