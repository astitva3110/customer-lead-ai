"""Phase 11 GPU runtime optimization tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from app.kb.embedding.batch_runner import BatchEmbeddingRunner
from app.kb.embedding.device import (
    EmbeddingDeviceError,
    get_device_info,
    is_cuda_oom_error,
    resolve_embedding_device,
)
from app.kb.vector.store import postgres_safe_text


class OomThenOkProvider:
    provider_name = "mock"
    model_name = "mock"
    model_revision = "rev"
    dimension = 1024

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail_next = True

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        if self.fail_next and len(texts) > 1:
            self.fail_next = False
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [[0.1] * 1024 for _ in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)


def test_default_batch_size_is_eight() -> None:
    from app.config import settings

    assert settings.embedding_batch_size == 8


def test_resolve_cpu_device() -> None:
    device = resolve_embedding_device("cpu")
    assert device.type == "cpu"


def test_cuda_requested_but_unavailable_raises() -> None:
    with patch("app.kb.embedding.device.cuda_is_available", return_value=False):
        with pytest.raises(EmbeddingDeviceError):
            resolve_embedding_device("cuda")


def test_cuda_device_when_available() -> None:
    with patch("app.kb.embedding.device.cuda_is_available", return_value=True):
        device = resolve_embedding_device("cuda")
        assert device.type == "cuda"


def test_is_cuda_oom_error() -> None:
    assert is_cuda_oom_error(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert is_cuda_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_cuda_oom_error(ValueError("other"))


def test_oom_batch_halving_retries_same_batch_without_skipping() -> None:
    provider = OomThenOkProvider()
    runner = BatchEmbeddingRunner(
        provider,
        initial_batch_size=4,
        device_label="cuda",
        gpu_name="mock-gpu",
    )
    texts = ["a", "b", "c", "d"]
    vectors = runner.embed_batch(texts)
    assert len(vectors) == 4
    assert runner.stats.oom_retries == 1
    assert 2 in runner.stats.batch_size_history
    assert provider.calls == [4, 2, 2]


def test_postgres_safe_text_does_not_alter_normal_content() -> None:
    original = "3.3 Securely package and ship to Sector 62, Noida"
    assert postgres_safe_text(original) == original


def test_postgres_safe_text_strips_nul_only_for_db() -> None:
    original = "before\x00after"
    sanitized = postgres_safe_text(original)
    assert "\x00" not in sanitized
    assert sanitized == "beforeafter"
    assert original != sanitized


def test_benchmark_mode_uses_benchmark_table_config() -> None:
    from app.kb.embedding.config import EmbeddingConfig
    from app.config import settings

    config = EmbeddingConfig.from_settings()
    assert config.vector_benchmark_table == settings.vector_benchmark_table
    assert config.vector_benchmark_table != config.vector_table


def test_get_device_info_cpu() -> None:
    info = get_device_info("cpu")
    assert info.resolved_device == "cpu"
    assert info.cuda_available == torch.cuda.is_available()


def test_deterministic_batch_runner_order_preserved() -> None:
    class SeqProvider:
        provider_name = "mock"
        model_name = "mock"
        model_revision = "rev"
        dimension = 1024

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] * 1024 for text in texts]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_batch(texts)

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return self.embed_batch(texts)

    runner = BatchEmbeddingRunner(SeqProvider(), initial_batch_size=2, device_label="cpu")
    texts = ["aa", "bbb", "cccc", "ddddd"]
    vectors = runner.embed_all(texts)
    assert [v[0] for v in vectors] == [2.0, 3.0, 4.0, 5.0]
