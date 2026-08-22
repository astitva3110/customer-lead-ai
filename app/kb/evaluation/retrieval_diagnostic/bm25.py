"""Deterministic read-only BM25 lexical retrieval diagnostic."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from app.kb.chunking.models import ProductionChunkRecord

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class Bm25Index:
    """In-memory BM25 over frozen production chunks."""

    def __init__(
        self,
        chunks: list[ProductionChunkRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._chunks = chunks
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._documents = [tokenize(chunk.content) for chunk in chunks]
        self._doc_lengths = [len(doc) for doc in self._documents]
        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        self._df: Counter[str] = Counter()
        for doc in self._documents:
            self._df.update(set(doc))
        self._n = len(self._documents)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def _score_document(self, query_terms: list[str], doc_index: int) -> float:
        doc = self._documents[doc_index]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = self._doc_lengths[doc_index]
        score = 0.0
        for term in query_terms:
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue
            idf = self._idf(term)
            denom = term_tf + self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
            score += idf * (term_tf * (self._k1 + 1.0)) / denom
        return score

    def search(self, query: str, *, top_k: int = 100) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scored: list[tuple[int, float]] = []
        for index in range(len(self._documents)):
            score = self._score_document(query_terms, index)
            if score > 0.0:
                scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(scored[:top_k], start=1):
            chunk = self._chunks[index]
            results.append(
                {
                    "rank": rank,
                    "score": round(score, 6),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "canonical_url": chunk.canonical_url,
                    "document_type": chunk.document_type.value,
                    "source_type": chunk.source_type.value,
                    "section_path": chunk.section_path,
                    "token_count": chunk.token_count,
                    "text": chunk.content,
                }
            )
        return results
