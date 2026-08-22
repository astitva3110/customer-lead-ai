#!/usr/bin/env python3
"""Index all V3.2 chunks into chunk_embeddings_v3_2 and compare in-memory vs PGVector.

Does not write chunk_embeddings or chunk_embeddings_phase12. Chat wiring unchanged.
Embeds the 127 V3.2 chunks once; in-memory ranking uses those same vectors.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.evaluation.catalog import CatalogItem, load_knowledge_catalog
from app.evaluation.evidence import row_matches_item
from app.kb.evaluation.corpus_registry import load_corpus
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import cosine_similarity, EmbeddingSession
from app.kb.ingestion.indexing import Phase12IndexIdentity, Phase12VectorStore
from app.kb.ingestion.models import DocumentRecord, Phase12ChunkRecord

FORBIDDEN_TABLES = {"chunk_embeddings", "chunk_embeddings_phase12"}
QUESTIONS_PATH = ROOT / "data/evaluations/earkart_kb_v3_1_comprehensive.json"
CONFIG_PATH = ROOT / "configs/retrieval/v3_2_pgvector.yaml"
TOP_K = 100


def section_label(path) -> str:
    if isinstance(path, list):
        return " / ".join(str(part) for part in path)
    return str(path or "")


def preview(text: str, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def identity_from_config(config: RetrievalConfig) -> Phase12IndexIdentity:
    return Phase12IndexIdentity(
        kb_dataset_version=config.kb_dataset_version,
        chunking_algorithm_version=config.chunking_algorithm_version,
        embedding_input_manifest=config.embedding_input_manifest,
        embedding_version=config.embedding_version,
        embedding_provider=settings.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_model_revision=config.embedding_revision,
        embedding_dimension=settings.embedding_dimension,
    )


def build_store(config: RetrievalConfig) -> Phase12VectorStore:
    table = config.vector_table
    if not table or table in FORBIDDEN_TABLES:
        raise SystemExit(f"refusing to write production table: {table!r}")
    store = Phase12VectorStore(table_name=table)
    store.identity = identity_from_config(config)
    store.ensure_schema()
    return store


def index_chunks(store: Phase12VectorStore, chunks: list[Phase12ChunkRecord], embeddings: list[list[float]]) -> int:
    grouped: dict[str, list[tuple[Phase12ChunkRecord, list[float]]]] = defaultdict(list)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        grouped[chunk.document_id].append((chunk, embedding))
    inserted = 0
    for document_id, pairs in grouped.items():
        doc_chunks = [item[0] for item in pairs]
        doc_embeddings = [item[1] for item in pairs]
        first = doc_chunks[0]
        title = (first.section_path[0] if first.section_path else document_id) or document_id
        record = DocumentRecord(
            document_id=document_id,
            document_version=first.document_version,
            filename=f"{document_id}.pdf",
            mime_type="application/pdf",
            file_hash=first.source_file_hash,
            document_type=first.document_type,
            created_at=datetime.now(timezone.utc),
            storage_path=f"eval/v3.2/{document_id}",
            title=title,
        )
        inserted += store.upsert_chunks(
            record=record,
            chunks=doc_chunks,
            embeddings=doc_embeddings,
            title=title,
            force=True,
        )
    return inserted


def memory_hits(query_vector, chunks: list[Phase12ChunkRecord], vectors: list[list[float]]) -> list[dict]:
    scored = [
        (cosine_similarity(query_vector, vector), chunk)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "rank": index,
            "chunk_id": chunk.chunk_id,
            "section_path": list(chunk.section_path),
            "text": chunk.content,
            "content_type": chunk.content_type,
            "similarity": round(sim, 6),
        }
        for index, (sim, chunk) in enumerate(scored[:TOP_K], start=1)
    ]


def pg_hits(store: Phase12VectorStore, query_vector) -> list[dict]:
    raw = store.search(query_vector, top_k=TOP_K, embedding_version=store.identity.embedding_version)
    return [
        {
            "rank": index,
            "chunk_id": item["chunk_id"],
            "section_path": list(item.get("section_path") or []),
            "text": item.get("content") or "",
            "content_type": item.get("split_method") or "",
            "similarity": item.get("similarity"),
        }
        for index, item in enumerate(raw, start=1)
    ]


def catalog_rank(hits: list[dict], item: CatalogItem) -> tuple[int | None, dict | None]:
    for hit in hits:
        row = {
            "chunk_id": hit["chunk_id"],
            "title": "",
            "section_path": hit["section_path"],
            "text": hit["text"],
        }
        if row_matches_item(row, item):
            return hit["rank"], hit
    return None, None


def describe(hit: dict | None) -> str:
    if hit is None:
        return "—"
    return f"{section_label(hit['section_path'])} :: {preview(str(hit.get('text') or ''))}"


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1

    def hit_at(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / n, 4)

    mrr = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    return {
        "n": len(ranks),
        "recall_at_1": hit_at(1),
        "recall_at_5": hit_at(5),
        "recall_at_10": hit_at(10),
        "mrr": round(sum(mrr) / n, 4),
    }


def main() -> int:
    config = RetrievalConfig.from_yaml(CONFIG_PATH)
    store = build_store(config)
    corpus = load_corpus("v3.2")
    chunks = corpus.chunks
    print(f"table={store.table_name} identity={store.identity.embedding_version} chunks={len(chunks)}", flush=True)
    if len(chunks) != 127:
        print(f"warning: expected 127 V3.2 chunks, got {len(chunks)}", flush=True)

    session = EmbeddingSession()
    print("embedding V3.2 chunks once", flush=True)
    vectors = session.embed_documents([chunk.embedding_input for chunk in chunks])
    inserted = index_chunks(store, chunks, vectors)
    stored = store.count(embedding_version=store.identity.embedding_version)
    print(f"upserted={inserted} stored={stored}", flush=True)

    catalog = [item for item in load_knowledge_catalog(QUESTIONS_PATH) if item.answerable]
    print(f"scoring {len(catalog)} answerable questions", flush=True)
    rows = []
    rank_agree = 0
    top1_agree = 0
    top10_agree = 0
    score_deltas: list[float] = []
    for item in catalog:
        query_vector = session.embed_queries([item.question])[0]
        mem = memory_hits(query_vector, chunks, vectors)
        pg = pg_hits(store, query_vector)
        mem_rank, mem_hit = catalog_rank(mem, item)
        pg_rank, pg_hit = catalog_rank(pg, item)
        mem_ids = [hit["chunk_id"] for hit in mem[:10]]
        pg_ids = [hit["chunk_id"] for hit in pg[:10]]
        if mem_rank == pg_rank:
            rank_agree += 1
        if mem_ids[:1] == pg_ids[:1]:
            top1_agree += 1
        if mem_ids == pg_ids:
            top10_agree += 1
        by_pg = {hit["chunk_id"]: hit["similarity"] for hit in pg}
        for hit in mem[:10]:
            other = by_pg.get(hit["chunk_id"])
            if other is not None and hit["similarity"] is not None:
                score_deltas.append(abs(float(hit["similarity"]) - float(other)))
        rows.append(
            {
                "id": item.question_id,
                "question": item.question,
                "category": item.category,
                "key": item.knowledge_key,
                "memory_rank": mem_rank,
                "pgvector_rank": pg_rank,
                "rank_equal": mem_rank == pg_rank,
                "top1_equal": mem_ids[:1] == pg_ids[:1],
                "memory_chunk": describe(mem_hit),
                "pgvector_chunk": describe(pg_hit),
                "memory_top1": mem_ids[0] if mem_ids else None,
                "pgvector_top1": pg_ids[0] if pg_ids else None,
            }
        )

    payload = {
        "production_tables_untouched": True,
        "table": store.table_name,
        "identity": store.identity.__dict__,
        "chunks_embedded": len(chunks),
        "rows_stored": stored,
        "metrics": {
            "memory": metrics([row["memory_rank"] for row in rows]),
            "pgvector": metrics([row["pgvector_rank"] for row in rows]),
        },
        "agreement": {
            "n": len(rows),
            "catalog_rank_equal": rank_agree,
            "top1_chunk_equal": top1_agree,
            "top10_list_equal": top10_agree,
            "max_abs_score_delta_top10": round(max(score_deltas), 6) if score_deltas else None,
            "mean_abs_score_delta_top10": round(sum(score_deltas) / len(score_deltas), 6) if score_deltas else None,
        },
        "questions": rows,
    }
    out = ROOT / "reports/mass_eval"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "v32_memory_vs_pgvector.json"
    txt_path = out / "v32_memory_vs_pgvector.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(format_text(payload), encoding="utf-8")
    print(format_text(payload))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


def format_text(payload: dict) -> str:
    mem = payload["metrics"]["memory"]
    pg = payload["metrics"]["pgvector"]
    agree = payload["agreement"]
    lines = [
        "V3_2_IN_MEMORY_VS_PGVECTOR",
        "production_tables_untouched: true",
        f"table: {payload['table']}",
        f"chunks_embedded: {payload['chunks_embedded']}",
        f"rows_stored: {payload['rows_stored']}",
        f"embedding_version: {payload['identity'].get('embedding_version')}",
        "",
        f"{'metric':<18} {'V3.2 memory':>14} {'V3.2 PGVector':>14}",
        f"{'Recall@1':<18} {mem['recall_at_1']:>14.4f} {pg['recall_at_1']:>14.4f}",
        f"{'Recall@5':<18} {mem['recall_at_5']:>14.4f} {pg['recall_at_5']:>14.4f}",
        f"{'Recall@10':<18} {mem['recall_at_10']:>14.4f} {pg['recall_at_10']:>14.4f}",
        f"{'MRR':<18} {mem['mrr']:>14.4f} {pg['mrr']:>14.4f}",
        "",
        "AGREEMENT",
        f"  catalog_rank_equal: {agree['catalog_rank_equal']}/{agree['n']}",
        f"  top1_chunk_equal: {agree['top1_chunk_equal']}/{agree['n']}",
        f"  top10_list_equal: {agree['top10_list_equal']}/{agree['n']}",
        f"  max_abs_score_delta_top10: {agree['max_abs_score_delta_top10']}",
        f"  mean_abs_score_delta_top10: {agree['mean_abs_score_delta_top10']}",
        "",
        "PER_QUESTION",
    ]
    for row in payload["questions"]:
        flag = "OK" if row["rank_equal"] and row["top1_equal"] else "DIFF"
        lines.append(
            f"  {row['id']} [{flag}] mem={row['memory_rank']} pg={row['pgvector_rank']} {row['question']}"
        )
        if flag != "OK":
            lines.append(f"    memory: {row['memory_chunk']}")
            lines.append(f"    pgvector: {row['pgvector_chunk']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
