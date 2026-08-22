from __future__ import annotations

from app.config import settings
from app.interfaces.providers.reranker import Reranker
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.reranker.lexical import LexicalOverlapReranker
from app.providers.reranker.passthrough import PassthroughReranker

AVAILABLE_RERANKERS = ("passthrough", "lexical_overlap", "cross_encoder")

RERANKER_NOTES = (
    "Available: passthrough, lexical_overlap, cross_encoder. "
    "Production default remains lexical_overlap until cross-encoder evaluation is promoted."
)


def create_reranker(provider: str | None = None) -> Reranker:
    name = (provider or settings.reranker_provider or "lexical_overlap").strip().lower()
    if name in {"lexical", "lexical_overlap"}:
        return LexicalOverlapReranker()
    if name == "passthrough":
        return PassthroughReranker()
    if name in {"cross_encoder", "cross-encoder"}:
        device = (settings.reranker_device or settings.embedding_device).strip() or "cpu"
        revision = settings.reranker_model_revision.strip() or None
        return CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            revision=revision,
        )
    raise ValueError(f"Unknown reranker {provider!r}. {RERANKER_NOTES}")
