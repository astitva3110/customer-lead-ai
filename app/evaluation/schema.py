"""Exploratory scenario dataset. Generator fields are hypotheses, not production GT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CATEGORIES = (
    "KNOWLEDGE_DIRECT",
    "KNOWLEDGE_PARAPHRASED",
    "KNOWLEDGE_FOLLOWUP",
    "PRODUCT_COMPARISON",
    "NOISY_QUERY",
    "CORPUS_GAP",
    "LEAD",
    "LEAD_WITH_KNOWLEDGE",
    "LEAD_PRODUCT_SWITCH",
    "SUPPORT",
    "SUPPORT_WITH_KNOWLEDGE",
    "SUPPORT_PRODUCT_SWITCH",
    "MIXED_INTENT",
    "VOLUNTEERED_CONTEXT",
    "GUARDRAIL",
    "PROMPT_INJECTION",
    "NONSENSE",
)

GOALS = ("NONE", "KNOWLEDGE", "LEAD", "SUPPORT")
INTENTS = (
    "KNOWLEDGE",
    "LEAD_INTENT",
    "SUPPORT_INTENT",
    "PROVIDE_INFORMATION",
    "CONTEXT_UPDATE",
    "CONFIRMATION",
    "ACTION",
    "MIXED",
    "GENERAL",
    "UNKNOWN",
    "GUARDRAIL",
    "NONSENSE",
)


class TurnMessage(BaseModel):
    """Scenario probe. expected_* and needs_* are hypotheses, not production GT."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(ge=1)
    role: str = "user"
    text: str = Field(min_length=1)
    expected_intent: str = "KNOWLEDGE"
    knowledge_keys: list[str] = Field(default_factory=list)
    answerable: bool | None = None
    needs_rag: bool = False
    needs_rewrite: bool = False
    expected_product: str | None = None
    expected_blocked: bool = False
    volunteered_fields: list[str] = Field(default_factory=list)
    explicit_product_correction: bool = False
    retrieval_bucket: str | None = None

    @field_validator("expected_intent")
    @classmethod
    def _intent(cls, value: str) -> str:
        if value not in INTENTS:
            raise ValueError(f"unknown expected_intent: {value}")
        return value


class ConversationExpected(BaseModel):
    """Scenario hypotheses for exploration. Not production ground truth."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool = True
    knowledge_keys: list[str] = Field(default_factory=list)
    goal: str = "KNOWLEDGE"
    rag_turns: list[int] = Field(default_factory=list)
    rewrite_turns: list[int] = Field(default_factory=list)
    tool_allowed_after_turn: int | None = None
    expected_product: str | None = None
    product_transitions: list[dict[str, Any]] = Field(default_factory=list)
    expected_lead_created: bool = False
    expected_ticket_created: bool = False
    volunteered_fields: list[str] = Field(default_factory=list)
    corpus_gap_turns: list[int] = Field(default_factory=list)
    guardrail_turns: list[int] = Field(default_factory=list)
    allow_duplicate_tools: bool = False

    @field_validator("goal")
    @classmethod
    def _goal(cls, value: str) -> str:
        if value not in GOALS:
            raise ValueError(f"unknown goal: {value}")
        return value


class GeneratedConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1)
    category: str
    language: str = "en"
    kind: str = "exploratory_scenario"
    messages: list[TurnMessage]
    expected: ConversationExpected

    @field_validator("category")
    @classmethod
    def _category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"unknown category: {value}")
        return value

    @model_validator(mode="after")
    def _turns_line_up(self) -> GeneratedConversation:
        if not self.messages:
            raise ValueError("conversation must have at least one message")
        turns = [item.turn for item in self.messages]
        if turns != list(range(1, len(self.messages) + 1)):
            raise ValueError(f"{self.conversation_id}: turns must be contiguous starting at 1")
        rag = {item.turn for item in self.messages if item.needs_rag}
        if set(self.expected.rag_turns) != rag:
            raise ValueError(f"{self.conversation_id}: expected.rag_turns must match needs_rag messages")
        return self


class GeneratedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    evaluator_version: str
    seed: int
    count: int
    generator_model: str
    corpus_version: str
    knowledge_catalog: str
    kind: str = "exploratory_scenario"
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    conversations: list[GeneratedConversation]

    @model_validator(mode="after")
    def _count_matches(self) -> GeneratedDataset:
        if self.count != len(self.conversations):
            raise ValueError(f"count {self.count} != conversations {len(self.conversations)}")
        ids = [item.conversation_id for item in self.conversations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate conversation_id values")
        return self


def conversation_id_for(index: int) -> str:
    return f"GEN-{index:06d}"


def load_generated_dataset(path: Path | str) -> GeneratedDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GeneratedDataset.model_validate(payload)


def dump_generated_dataset(dataset: GeneratedDataset, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return target


def validate_dataset_payload(payload: dict[str, Any] | GeneratedDataset) -> GeneratedDataset:
    if isinstance(payload, GeneratedDataset):
        return payload
    return GeneratedDataset.model_validate(payload)
