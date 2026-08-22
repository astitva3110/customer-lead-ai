#!/usr/bin/env python3
"""Controlled comparison: V3.1 in-memory vs V3.2 in-memory vs Phase-12 PGVector.

Same verified catalog questions. Same query encoder. Production index is not re-embedded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.catalog import CatalogItem, load_knowledge_catalog
from app.evaluation.evidence import row_matches_item
from app.kb.evaluation.corpus_registry import load_corpus
from app.kb.evaluation.search.backends import cosine_similarity, EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore
from app.kb.ingestion.models import Phase12ChunkRecord

QUESTIONS_PATH = ROOT / "data/evaluations/earkart_kb_v3_1_comprehensive.json"
PHASE12_TABLE = "chunk_embeddings_phase12"
PHASE12_VERSION = "qwen_Qwen3-Embedding-0.6B_phase12_v1"
TOP_K = 100
CATEGORY_KEYS = ("PRODUCT", "HEARING_AID", "POLICY", "SUMMARY")


def section_label(path) -> str:
    if isinstance(path, list):
        return " / ".join(str(part) for part in path)
    return str(path or "")


def preview(text: str, limit: int = 180) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def chunk_row(chunk: Phase12ChunkRecord, similarity: float) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "title": "",
        "section_path": list(chunk.section_path),
        "text": chunk.content,
        "content_type": chunk.content_type,
        "similarity": round(similarity, 6),
        "token_count": chunk.token_count,
    }


def store_row(item: dict) -> dict:
    return {
        "chunk_id": item.get("chunk_id"),
        "title": item.get("title") or "",
        "section_path": list(item.get("section_path") or []),
        "text": item.get("content") or "",
        "content_type": item.get("split_method") or "",
        "similarity": item.get("similarity"),
        "token_count": None,
    }


def match_rank(rows: list[dict], item: CatalogItem) -> tuple[int | None, dict | None]:
    for index, row in enumerate(rows, start=1):
        if row_matches_item(row, item):
            return index, row
    return None, None


def describe(row: dict | None) -> str:
    if row is None:
        return "—"
    section = section_label(row.get("section_path"))
    ctype = row.get("content_type") or ""
    extra = f" [{ctype}]" if ctype else ""
    return f"{section}{extra} :: {preview(str(row.get('text') or ''))}"


def rank_in_memory(query_vector, chunks: list[Phase12ChunkRecord], vectors: list[list[float]], item: CatalogItem):
    scored = [
        (cosine_similarity(query_vector, vector), chunk)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    rows = [chunk_row(chunk, sim) for sim, chunk in scored[:TOP_K]]
    rank, matched = match_rank(rows, item)
    return {
        "rank": rank,
        "score": None if matched is None else matched.get("similarity"),
        "chunk": describe(matched),
        "chunk_id": None if matched is None else matched.get("chunk_id"),
        "top1": describe(rows[0]) if rows else "—",
    }


def rank_production(store: Phase12VectorStore, query_vector, item: CatalogItem):
    raw = store.search(query_vector, top_k=TOP_K, embedding_version=PHASE12_VERSION)
    rows = [store_row(item_raw) for item_raw in raw]
    rank, matched = match_rank(rows, item)
    return {
        "rank": rank,
        "score": None if matched is None else matched.get("similarity"),
        "chunk": describe(matched),
        "chunk_id": None if matched is None else matched.get("chunk_id"),
        "top1": describe(rows[0]) if rows else "—",
    }


def metrics_from_ranks(rows: list[dict], key: str) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    n = len(answerable) or 1

    def hit(rank, k):
        return rank is not None and rank <= k

    recall = {
        f"recall_at_{k}": round(sum(1 for row in answerable if hit(row[key]["rank"], k)) / n, 4)
        for k in (1, 5, 10)
    }
    mrr_vals = [0.0 if row[key]["rank"] is None else 1.0 / row[key]["rank"] for row in answerable]
    recall["mrr"] = round(sum(mrr_vals) / n, 4)
    recall["n"] = len(answerable)
    for category in CATEGORY_KEYS:
        subset = [row for row in answerable if row["category"] == category]
        label = {
            "PRODUCT": "product_at_5",
            "HEARING_AID": "hearing_aid_at_5",
            "POLICY": "policy_at_5",
            "SUMMARY": "summary_at_5",
        }[category]
        denom = len(subset) or 1
        recall[label] = round(sum(1 for row in subset if hit(row[key]["rank"], 5)) / denom, 4)
        recall[f"{label}_n"] = len(subset)
    return recall


def main() -> int:
    catalog = {item.question_id: item for item in load_knowledge_catalog(QUESTIONS_PATH)}
    session = EmbeddingSession()
    v31 = load_corpus("v3.1")
    v32 = load_corpus("v3.2")
    print(f"embedding v3.1 chunks n={len(v31.chunks)}", flush=True)
    v31_vectors = session.embed_documents([chunk.embedding_input for chunk in v31.chunks])
    print(f"embedding v3.2 chunks n={len(v32.chunks)}", flush=True)
    v32_vectors = session.embed_documents([chunk.embedding_input for chunk in v32.chunks])
    store = Phase12VectorStore(table_name=PHASE12_TABLE)

    questions = sorted(catalog.values(), key=lambda item: item.question_id)
    print(f"scoring {len(questions)} questions", flush=True)
    rows = []
    for item in questions:
        query_vector = session.embed_queries([item.question])[0]
        rows.append(
            {
                "id": item.question_id,
                "question": item.question,
                "category": item.category,
                "key": item.knowledge_key,
                "answerable": item.answerable,
                "v3_1": rank_in_memory(query_vector, v31.chunks, v31_vectors, item),
                "v3_2": rank_in_memory(query_vector, v32.chunks, v32_vectors, item),
                "production": rank_production(store, query_vector, item),
            }
        )

    payload = {
        "ground_truth": "catalog anchors (section_path_contains + content_patterns)",
        "query_encoder": "current Qwen3-Embedding-0.6B, same vectors for all three indexes",
        "production_reindex": False,
        "corpus": {
            "v3_1_chunks": len(v31.chunks),
            "v3_2_chunks": len(v32.chunks),
            "production_table": PHASE12_TABLE,
            "production_embedding_version": PHASE12_VERSION,
        },
        "metrics": {
            "v3_1": metrics_from_ranks(rows, "v3_1"),
            "v3_2": metrics_from_ranks(rows, "v3_2"),
            "production": metrics_from_ranks(rows, "production"),
        },
        "questions": rows,
    }
    out = ROOT / "reports/mass_eval"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "corpus_index_comparison.json"
    txt_path = out / "corpus_index_comparison.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(format_text(payload), encoding="utf-8")
    print(format_text(payload))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


def format_text(payload: dict) -> str:
    m = payload["metrics"]
    lines = [
        "CORPUS_INDEX_COMPARISON",
        "questions: earkart_kb_v3_1_comprehensive.json",
        "ground_truth: catalog anchors (same matcher on all three indexes)",
        "production_reindex: false",
        "",
        f"{'metric':<18} {'V3.1':>10} {'V3.2':>10} {'Production':>12}",
        f"{'Recall@1':<18} {m['v3_1']['recall_at_1']:>10.4f} {m['v3_2']['recall_at_1']:>10.4f} {m['production']['recall_at_1']:>12.4f}",
        f"{'Recall@5':<18} {m['v3_1']['recall_at_5']:>10.4f} {m['v3_2']['recall_at_5']:>10.4f} {m['production']['recall_at_5']:>12.4f}",
        f"{'Recall@10':<18} {m['v3_1']['recall_at_10']:>10.4f} {m['v3_2']['recall_at_10']:>10.4f} {m['production']['recall_at_10']:>12.4f}",
        f"{'MRR':<18} {m['v3_1']['mrr']:>10.4f} {m['v3_2']['mrr']:>10.4f} {m['production']['mrr']:>12.4f}",
        f"{'Product@5':<18} {m['v3_1']['product_at_5']:>10.4f} {m['v3_2']['product_at_5']:>10.4f} {m['production']['product_at_5']:>12.4f}",
        f"{'HearingAid@5':<18} {m['v3_1']['hearing_aid_at_5']:>10.4f} {m['v3_2']['hearing_aid_at_5']:>10.4f} {m['production']['hearing_aid_at_5']:>12.4f}",
        f"{'Policy@5':<18} {m['v3_1']['policy_at_5']:>10.4f} {m['v3_2']['policy_at_5']:>10.4f} {m['production']['policy_at_5']:>12.4f}",
        f"{'Summary@5':<18} {m['v3_1']['summary_at_5']:>10.4f} {m['v3_2']['summary_at_5']:>10.4f} {m['production']['summary_at_5']:>12.4f}",
        "",
        "PER_QUESTION",
    ]
    for row in payload["questions"]:
        gap = "" if row["answerable"] else " CORPUS_GAP"
        lines.append(f"  {row['id']} {row['question']}{gap}")
        lines.append(
            f"    rank  V3.1={row['v3_1']['rank']}  V3.2={row['v3_2']['rank']}  Production={row['production']['rank']}"
        )
        lines.append(f"    V3.1 chunk: {row['v3_1']['chunk']}")
        lines.append(f"    V3.2 chunk: {row['v3_2']['chunk']}")
        lines.append(f"    Production chunk: {row['production']['chunk']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
