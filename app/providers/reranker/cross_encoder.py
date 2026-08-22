"""Local Hugging Face cross-encoder reranker.

Selected first-benchmark model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``.

Score semantics
---------------
The model is a sequence-classification head that emits a **single relevance
logit** per (query, chunk) pair. Higher is more relevant. Typical raw range is
roughly -10 to +15; it is **not** a probability and is **not** comparable to
vector cosine or lexical coverage.

This reranker stores:

* ``rerank_score`` = sigmoid(logit) in (0, 1), monotonic with the logit so
  ranking is unchanged. ``RERANKER_MIN_SCORE=0.0`` therefore does not drop
  candidates during ranking evaluation.
* ``metadata['rerank_logit']`` = the raw logit, for later threshold work.

Do not treat ``rerank_score = 0.8`` as the same meaning as lexical coverage 0.8.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.retrieval.models import RankedCandidate, RetrievalCandidate
from app.kb.embedding.device import resolve_embedding_device

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]
LoaderFn = Callable[[str, str, str | None], tuple[Any, Any, torch.device]]

_LOADED_MODELS: dict[tuple[str, str, str], tuple[Any, Any, torch.device]] = {}


def reset_cross_encoder_cache() -> None:
    _LOADED_MODELS.clear()


def load_cross_encoder_model(
    model_name: str,
    device: str,
    revision: str | None = None,
) -> tuple[Any, Any, torch.device]:
    resolved = resolve_embedding_device(device)
    cache_key = (model_name, str(resolved), revision or "")
    cached = _LOADED_MODELS.get(cache_key)
    if cached is not None:
        return cached

    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(resolved)
    packed = (tokenizer, model, resolved)
    _LOADED_MODELS[cache_key] = packed
    return packed


class CrossEncoderReranker:
    """Batched (query, chunk) cross-encoder. Lazy-loads the HF model once."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        revision: str | None = None,
        score_fn: ScoreFn | None = None,
        loader: LoaderFn | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self._device_name = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._revision = revision
        self._score_fn = score_fn
        self._loader = loader or load_cross_encoder_model
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None
        self.load_count = 0

    def _ensure_loaded(self) -> None:
        if self._score_fn is not None:
            return
        if self._model is not None:
            return
        tokenizer, model, device = self._loader(self.model_name, self._device_name, self._revision)
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self.load_count += 1

    @property
    def parameter_device(self) -> str | None:
        if self._model is None:
            return None
        return str(next(self._model.parameters()).device)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RankedCandidate]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        if self._score_fn is not None:
            probs = self._score_fn(pairs)
            logits = [None] * len(probs)
        else:
            self._ensure_loaded()
            logits, probs = self._score_pairs(pairs)
        if len(probs) != len(candidates):
            raise ValueError("cross-encoder returned a different number of scores than candidates")

        scored: list[RankedCandidate] = []
        for candidate, logit, prob in zip(candidates, logits, probs, strict=True):
            metadata = dict(candidate.metadata)
            if logit is not None:
                metadata["rerank_logit"] = round(float(logit), 6)
            scored.append(
                replace(
                    candidate,
                    rerank_score=round(float(prob), 6),
                    metadata=metadata,
                )
            )
        scored.sort(
            key=lambda item: (
                item.rerank_score or 0.0,
                item.vector_score or 0.0,
                item.keyword_score or 0.0,
            ),
            reverse=True,
        )
        return scored

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> tuple[list[float], list[float]]:
        assert self._tokenizer is not None and self._model is not None and self._device is not None
        logits_out: list[float] = []
        probs_out: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            encoded = self._tokenizer(
                queries,
                texts,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                raw = self._model(**encoded).logits.view(-1)
                probs = torch.sigmoid(raw)
            logits_out.extend(float(value) for value in raw.detach().cpu())
            probs_out.extend(float(value) for value in probs.detach().cpu())
        return logits_out, probs_out
