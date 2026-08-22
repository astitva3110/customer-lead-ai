from pathlib import Path

from app.config import settings
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.reranker.factory import create_reranker
from app.providers.reranker.lexical import LexicalOverlapReranker
from app.providers.reranker.passthrough import PassthroughReranker
from app.providers.retrieval.factory import build_chat_retriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.retrieval.service import RetrievalService


def test_create_reranker_default_is_lexical() -> None:
    assert isinstance(create_reranker("lexical_overlap"), LexicalOverlapReranker)
    assert isinstance(create_reranker("passthrough"), PassthroughReranker)


def test_create_reranker_cross_encoder_is_lazy() -> None:
    reranker = create_reranker("cross_encoder")
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.load_count == 0


def test_reranker_device_is_independent_of_embedding_device(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_device", "cpu")
    monkeypatch.setattr(settings, "reranker_device", "cuda")
    reranker = create_reranker("cross_encoder")
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._device_name == "cuda"
    assert reranker.load_count == 0

    monkeypatch.setattr(settings, "reranker_device", "")
    monkeypatch.setattr(settings, "embedding_device", "cpu")
    cpu_fallback = create_reranker("cross_encoder")
    assert cpu_fallback._device_name == "cpu"


def test_hybrid_disabled_returns_retrieval_service(monkeypatch) -> None:
    class _BoomBackend:
        def search(self, query, *, top_k):
            return []

    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", False)

    def fake_from_config(cls, config, **kwargs):
        del config, kwargs
        return RetrievalService(_BoomBackend())

    monkeypatch.setattr(RetrievalService, "from_config", classmethod(fake_from_config))
    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))
    retriever = build_chat_retriever(config)
    assert isinstance(retriever, RetrievalService)
