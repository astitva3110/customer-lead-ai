"""Local Qwen3 embedding provider."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.device import clear_cuda_cache, resolve_embedding_device


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _format_query(instruction: str, query: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"


class QwenLocalEmbeddingProvider:
    """Qwen/Qwen3-Embedding-0.6B local inference via transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._device = resolve_embedding_device(config.device)
        use_fp16 = self._device.type == "cuda"
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.model_revision,
            trust_remote_code=True,
            padding_side="left",
        )
        self._model = AutoModel.from_pretrained(
            config.model,
            revision=config.model_revision,
            trust_remote_code=True,
            dtype=torch.float16 if use_fp16 else torch.float32,
        )
        self._model.eval()
        self._model.to(self._device)

    @property
    def provider_name(self) -> str:
        return "qwen_local"

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def model_revision(self) -> str:
        return self._config.model_revision

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def resolved_device(self) -> str:
        return str(self._device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.embed_batch(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        instructed = [_format_query(self._config.query_instruction, text) for text in texts]
        return self.embed_batch(instructed)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if self._config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            vectors = embeddings.float().cpu().numpy()
        del inputs, outputs, embeddings
        if self._device.type == "cuda":
            clear_cuda_cache()
        return [self._validate_vector(row) for row in vectors]

    def _validate_vector(self, row: np.ndarray) -> list[float]:
        if row.shape[0] != self._config.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._config.dimension}, got {row.shape[0]}"
            )
        if not np.all(np.isfinite(row)):
            raise ValueError("Embedding contains NaN or infinite values")
        if self._config.normalize:
            norm = float(np.linalg.norm(row))
            if not np.isclose(norm, 1.0, atol=1e-3):
                raise ValueError(f"Expected L2-normalized vector, got norm={norm}")
        return row.astype(float).tolist()
