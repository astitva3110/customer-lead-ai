"""Load mass-conversation evaluation configuration. Version is data, not code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.evaluation.version import EVALUATOR_VERSION

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "evaluation" / "mass_conversation.yaml"

DEFAULT_CATEGORIES: dict[str, float] = {
    "KNOWLEDGE_DIRECT": 0.10,
    "KNOWLEDGE_PARAPHRASED": 0.08,
    "KNOWLEDGE_FOLLOWUP": 0.08,
    "PRODUCT_COMPARISON": 0.05,
    "NOISY_QUERY": 0.05,
    "CORPUS_GAP": 0.05,
    "LEAD": 0.10,
    "LEAD_WITH_KNOWLEDGE": 0.10,
    "LEAD_PRODUCT_SWITCH": 0.05,
    "SUPPORT": 0.08,
    "SUPPORT_WITH_KNOWLEDGE": 0.08,
    "SUPPORT_PRODUCT_SWITCH": 0.03,
    "MIXED_INTENT": 0.05,
    "VOLUNTEERED_CONTEXT": 0.03,
    "GUARDRAIL": 0.03,
    "PROMPT_INJECTION": 0.02,
    "NONSENSE": 0.03,
}


@dataclass(frozen=True)
class MassEvalConfig:
    evaluator_version: str = EVALUATOR_VERSION
    dataset_version: str = "mass-v1"
    corpus_version: str = "v3.2"
    knowledge_catalog: str = "data/evaluations/earkart_kb_v3_1_comprehensive.json"
    generator_model: str = "qwen3:1.7b"
    ollama_endpoint: str = "http://127.0.0.1:11434"
    use_llm: bool = True
    generator_temperature: float = 0.7
    generator_timeout_seconds: float = 60.0
    count: int = 1000
    seed: int = 42
    min_turns: int = 3
    max_turns: int = 12
    language_distribution: dict[str, float] = field(default_factory=lambda: {"en": 0.9, "hi": 0.1})
    category_distribution: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CATEGORIES))
    concurrency: int = 4
    timeout_seconds: float = 120.0
    retry_count: int = 1
    resume: bool = True
    trace_enabled: bool = True
    save_full_traces: bool = True
    chat_trace_retrieval_top_k: int = 10
    output_dir: str = "reports/mass_eval"
    generated_output: str = "data/evaluations/generated/v1.json"

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> MassEvalConfig:
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MassEvalConfig:
        payload = dict(data or {})
        categories = dict(payload.get("category_distribution") or DEFAULT_CATEGORIES)
        languages = dict(payload.get("language_distribution") or {"en": 0.9, "hi": 0.1})
        return cls(
            evaluator_version=str(payload.get("evaluator_version") or EVALUATOR_VERSION),
            dataset_version=str(payload.get("dataset_version") or "mass-v1"),
            corpus_version=str(payload.get("corpus_version") or "v3.2"),
            knowledge_catalog=str(
                payload.get("knowledge_catalog")
                or "data/evaluations/earkart_kb_v3_1_comprehensive.json"
            ),
            generator_model=str(payload.get("generator_model") or "qwen3:1.7b"),
            ollama_endpoint=str(payload.get("ollama_endpoint") or "http://127.0.0.1:11434"),
            use_llm=bool(payload.get("use_llm", True)),
            generator_temperature=float(payload.get("generator_temperature", 0.7)),
            generator_timeout_seconds=float(payload.get("generator_timeout_seconds", 60)),
            count=int(payload.get("count", 1000)),
            seed=int(payload.get("seed", 42)),
            min_turns=int(payload.get("min_turns", 3)),
            max_turns=int(payload.get("max_turns", 12)),
            language_distribution=languages,
            category_distribution=categories,
            concurrency=max(1, int(payload.get("concurrency", 4))),
            timeout_seconds=float(payload.get("timeout_seconds", 120)),
            retry_count=max(0, int(payload.get("retry_count", 1))),
            resume=bool(payload.get("resume", True)),
            trace_enabled=bool(payload.get("trace_enabled", True)),
            save_full_traces=bool(payload.get("save_full_traces", True)),
            chat_trace_retrieval_top_k=int(payload.get("chat_trace_retrieval_top_k", 10)),
            output_dir=str(payload.get("output_dir") or "reports/mass_eval"),
            generated_output=str(payload.get("generated_output") or "data/evaluations/generated/v1.json"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_version": self.evaluator_version,
            "dataset_version": self.dataset_version,
            "corpus_version": self.corpus_version,
            "knowledge_catalog": self.knowledge_catalog,
            "generator_model": self.generator_model,
            "ollama_endpoint": self.ollama_endpoint,
            "use_llm": self.use_llm,
            "generator_temperature": self.generator_temperature,
            "generator_timeout_seconds": self.generator_timeout_seconds,
            "count": self.count,
            "seed": self.seed,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "language_distribution": dict(self.language_distribution),
            "category_distribution": dict(self.category_distribution),
            "concurrency": self.concurrency,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "resume": self.resume,
            "trace_enabled": self.trace_enabled,
            "save_full_traces": self.save_full_traces,
            "chat_trace_retrieval_top_k": self.chat_trace_retrieval_top_k,
            "output_dir": self.output_dir,
            "generated_output": self.generated_output,
        }

    def catalog_path(self) -> Path:
        path = Path(self.knowledge_catalog)
        return path if path.is_absolute() else ROOT / path

    def output_path(self) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else ROOT / path

    def generated_path(self) -> Path:
        path = Path(self.generated_output)
        return path if path.is_absolute() else ROOT / path


def load_mass_eval_config(path: Path | str | None = None) -> MassEvalConfig:
    return MassEvalConfig.from_yaml(path)
