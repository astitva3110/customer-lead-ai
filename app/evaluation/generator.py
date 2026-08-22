"""Offline exploratory scenario generator. Not part of production /chat."""

from __future__ import annotations

import json
import random
from typing import Any

import httpx

from app.evaluation.catalog import CatalogItem, load_knowledge_catalog
from app.evaluation.config import MassEvalConfig
from app.evaluation.schema import ConversationExpected, GeneratedConversation, GeneratedDataset, TurnMessage
from app.evaluation.templates import build_conversation
from app.evaluation.variation import allocate_counts, weighted_choice
from app.helpers.generation_json import extract_json_object


def generate_dataset(config: MassEvalConfig, *, count: int | None = None, use_llm: bool | None = None) -> GeneratedDataset:
    total = int(count if count is not None else config.count)
    if total < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(config.seed)
    catalog = load_knowledge_catalog(config.catalog_path())
    categories = allocate_counts(total, config.category_distribution, rng)
    conversations: list[GeneratedConversation] = []
    llm = use_llm if use_llm is not None else config.use_llm
    for index, category in enumerate(categories, start=1):
        language = weighted_choice(rng, config.language_distribution)
        conversation = build_conversation(
            index=index,
            category=category,
            language=language,
            rng=rng,
            catalog=catalog,
            min_turns=config.min_turns,
            max_turns=config.max_turns,
        )
        if llm:
            conversation = maybe_paraphrase(conversation, config, rng)
        conversations.append(_keep_verified_catalog_keys(conversation, catalog))
    return GeneratedDataset(
        dataset_version=config.dataset_version,
        evaluator_version=config.evaluator_version,
        seed=config.seed,
        count=len(conversations),
        generator_model=config.generator_model if llm else "template",
        corpus_version=config.corpus_version,
        knowledge_catalog=config.knowledge_catalog,
        kind="exploratory_scenario",
        config_snapshot=config.to_dict(),
        conversations=conversations,
    )


def maybe_paraphrase(
    conversation: GeneratedConversation,
    config: MassEvalConfig,
    rng: random.Random,
) -> GeneratedConversation:
    try:
        rewritten = paraphrase_with_ollama(conversation, config)
    except Exception:
        return conversation
    if rewritten is None:
        return conversation
    messages: list[TurnMessage] = []
    for original, text in zip(conversation.messages, rewritten):
        payload = original.model_dump()
        payload["text"] = text
        messages.append(TurnMessage.model_validate(payload))
    return conversation.model_copy(update={"messages": messages})


def paraphrase_with_ollama(conversation: GeneratedConversation, config: MassEvalConfig) -> list[str] | None:
    originals = [item.text for item in conversation.messages]
    user = json.dumps(
        {
            "category": conversation.category,
            "messages": originals,
            "instruction": (
                "Rewrite each user message in natural language. Keep product names, "
                "phone numbers, cities, and names unchanged. Keep the same number of messages. "
                "Do not add new facts. Return JSON {\"messages\": [\"...\"]}."
            ),
        },
        ensure_ascii=False,
    )
    payload = _ollama_generate(config, user)
    parsed = extract_json_object(payload) or {}
    messages = parsed.get("messages")
    if not isinstance(messages, list) or len(messages) != len(originals):
        return None
    cleaned = [str(item).strip() for item in messages]
    if any(not item for item in cleaned):
        return None
    return cleaned


def _ollama_generate(config: MassEvalConfig, prompt: str) -> str:
    url = config.ollama_endpoint.rstrip("/") + "/api/generate"
    body: dict[str, Any] = {
        "model": config.generator_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": config.generator_temperature, "seed": config.seed},
    }
    response = httpx.post(url, json=body, timeout=config.generator_timeout_seconds)
    response.raise_for_status()
    data = response.json()
    return str(data.get("response") or "")


def _keep_verified_catalog_keys(
    conversation: GeneratedConversation,
    catalog: list[CatalogItem],
) -> GeneratedConversation:
    by_key = {item.knowledge_key: item for item in catalog}
    messages: list[TurnMessage] = []
    for message in conversation.messages:
        payload = message.model_dump()
        keys = [key for key in message.knowledge_keys if key in by_key]
        payload["knowledge_keys"] = keys
        if keys:
            items = [by_key[key] for key in keys]
            if all(not item.answerable for item in items):
                payload["answerable"] = False
            elif any(item.answerable for item in items):
                payload["answerable"] = True
        messages.append(TurnMessage.model_validate(payload))
    expected = conversation.expected.model_dump()
    expected["knowledge_keys"] = [key for key in expected.get("knowledge_keys") or [] if key in by_key]
    expected["rag_turns"] = [item.turn for item in messages if item.needs_rag]
    expected["corpus_gap_turns"] = [
        item.turn for item in messages if item.answerable is False and item.needs_rag
    ]
    return conversation.model_copy(
        update={
            "kind": "exploratory_scenario",
            "messages": messages,
            "expected": ConversationExpected.model_validate(expected),
        }
    )
