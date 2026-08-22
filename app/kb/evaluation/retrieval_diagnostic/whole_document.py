"""In-memory whole-document embedding diagnostics (never persisted)."""

from __future__ import annotations

from typing import Any

from app.kb.chunking.models import ProductionChunkRecord
from app.kb.embedding.batch_runner import BatchEmbeddingRunner
from app.kb.embedding.provider import EmbeddingProvider


def build_document_texts(chunks: list[ProductionChunkRecord]) -> dict[str, str]:
    by_document: dict[str, list[ProductionChunkRecord]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)
    texts: dict[str, str] = {}
    for document_id, doc_chunks in by_document.items():
        ordered = sorted(doc_chunks, key=lambda item: item.chunk_index)
        texts[document_id] = "\n\n".join(chunk.content for chunk in ordered)
    return texts


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return round(sum(a * b for a, b in zip(left, right)), 6)


def document_best_chunk_similarities(vector_hits: list[dict[str, Any]]) -> dict[str, float]:
    """Proxy document rank using best chunk similarity already retrieved."""
    best: dict[str, float] = {}
    for hit in vector_hits:
        document_id = hit.get("document_id")
        if not document_id:
            continue
        similarity = float(hit.get("similarity") or 0.0)
        best[document_id] = max(best.get(document_id, 0.0), similarity)
    return best


def rank_document_by_chunk_proxy(
    document_id: str,
    vector_hits: list[dict[str, Any]],
) -> int | None:
    best = document_best_chunk_similarities(vector_hits)
    if document_id not in best:
        return None
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    rank_by_id = {doc_id: index + 1 for index, (doc_id, _sim) in enumerate(ranked)}
    return rank_by_id.get(document_id)


class WholeDocumentIndex:
    """Embeds only expected documents in memory; uses chunk hits for comparison rank."""

    def __init__(self, *, provider: EmbeddingProvider, document_texts: dict[str, str]) -> None:
        self._provider = provider
        self._document_texts = document_texts
        self._vector_cache: dict[str, list[float]] = {}

    def _embed_expected_documents(self, document_ids: list[str]) -> None:
        missing = [
            doc_id
            for doc_id in document_ids
            if doc_id not in self._vector_cache and doc_id in self._document_texts
        ]
        if not missing:
            return
        device = getattr(getattr(self._provider, "_config", None), "device", "cpu")
        runner = BatchEmbeddingRunner(
            self._provider,
            initial_batch_size=1,
            device_label=str(device),
        )
        vectors = runner.embed_all([self._document_texts[doc_id] for doc_id in missing])
        for doc_id, vector in zip(missing, vectors):
            self._vector_cache[doc_id] = vector

    def analyze(
        self,
        query_vector: list[float],
        expected_document_ids: list[str],
        *,
        vector_hits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not expected_document_ids:
            return {
                "available": False,
                "expected_document_id": None,
                "similarity": None,
                "rank_among_documents": None,
            }

        primary_document_id = expected_document_ids[0]
        if primary_document_id not in self._document_texts:
            return {
                "available": False,
                "expected_document_id": primary_document_id,
                "similarity": None,
                "rank_among_documents": None,
            }

        self._embed_expected_documents(expected_document_ids)
        primary_vector = self._vector_cache.get(primary_document_id)
        if primary_vector is None:
            return {
                "available": False,
                "expected_document_id": primary_document_id,
                "similarity": None,
                "rank_among_documents": None,
            }

        primary_similarity = cosine_similarity(query_vector, primary_vector)
        rank_among_documents = rank_document_by_chunk_proxy(primary_document_id, vector_hits or [])

        return {
            "available": True,
            "expected_document_id": primary_document_id,
            "similarity": primary_similarity,
            "rank_among_documents": rank_among_documents,
            "rank_method": "best_chunk_in_top_100",
        }
