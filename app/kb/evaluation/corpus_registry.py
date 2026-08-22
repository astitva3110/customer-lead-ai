"""Corpus version registry — version is data, loader is registered once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.kb.ingestion.models import Phase12ChunkRecord


@dataclass
class LoadedCorpus:
    version: str
    chunks: list[Phase12ChunkRecord]
    source_pdfs: list[str]
    metadata: dict


CorpusLoader = Callable[[], LoadedCorpus]

_REGISTRY: dict[str, CorpusLoader] = {}


def register_corpus(version: str, loader: CorpusLoader) -> None:
    _REGISTRY[version] = loader


def get_corpus_loader(version: str) -> CorpusLoader:
    if version not in _REGISTRY:
        raise KeyError(f"Unknown corpus version: {version!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[version]


def load_corpus(version: str) -> LoadedCorpus:
    return get_corpus_loader(version)()


def registered_versions() -> list[str]:
    return sorted(_REGISTRY)


def _bootstrap_registry() -> None:
    if _REGISTRY:
        return

    def _load_v2() -> LoadedCorpus:
        from app.kb.embedding.phase13_corpus import load_phase13_validated_corpus

        corpus = load_phase13_validated_corpus()
        return LoadedCorpus(
            version="v2",
            chunks=corpus.all_chunks,
            source_pdfs=corpus.source_pdfs,
            metadata={"chunk_count": corpus.chunk_count},
        )

    def _load_v3() -> LoadedCorpus:
        from app.kb.ingestion.phase16_corpus import load_phase16_v3_corpus

        corpus = load_phase16_v3_corpus()
        return LoadedCorpus(
            version="v3",
            chunks=corpus.all_chunks,
            source_pdfs=corpus.source_pdfs,
            metadata={"chunk_count": corpus.chunk_count},
        )

    def _load_v3_1() -> LoadedCorpus:
        from app.kb.ingestion.phase16_corpus_v3_1 import load_phase16_v3_1_corpus

        corpus = load_phase16_v3_1_corpus()
        return LoadedCorpus(
            version="v3.1",
            chunks=corpus.all_chunks,
            source_pdfs=corpus.source_pdfs,
            metadata={
                "chunk_count": corpus.chunk_count,
                "atomic_chunks": sum(
                    1 for chunk in corpus.all_chunks if chunk.content_type != "section_summary"
                ),
                "summary_chunks": sum(
                    1 for chunk in corpus.all_chunks if chunk.content_type == "section_summary"
                ),
            },
        )

    def _load_v3_2() -> LoadedCorpus:
        from app.kb.ingestion.phase16_corpus_v3_2 import build_v3_2_integrity_report, load_phase16_v3_2_corpus

        corpus = load_phase16_v3_2_corpus()
        integrity = build_v3_2_integrity_report()
        return LoadedCorpus(
            version="v3.2",
            chunks=corpus.all_chunks,
            source_pdfs=corpus.source_pdfs,
            metadata={
                "parent_version": "v3.1",
                "chunk_count": corpus.chunk_count,
                "atomic_chunks": integrity.v3_1_atomic_chunks,
                "summary_chunks": integrity.v3_1_summary_chunks + integrity.v3_2_new_summary_chunks,
                "new_summary_chunks": integrity.v3_2_new_summary_chunks,
                "chunking_algorithm_version": "phase16.2.0",
            },
        )

    register_corpus("v2", _load_v2)
    register_corpus("v3", _load_v3)
    register_corpus("v3.1", _load_v3_1)
    register_corpus("v3.2", _load_v3_2)


_bootstrap_registry()
