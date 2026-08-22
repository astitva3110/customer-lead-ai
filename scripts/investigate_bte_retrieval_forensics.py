#!/usr/bin/env python3
"""Phase 24.1 — BTE candidate retrieval forensics against live Phase 12 PGVector.

Read-only. Does not re-index, upsert, or change production retrieval.
A single-document identity probe embeds stored/reconstructed texts only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import select, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.conversation.models import ConversationState
from app.services.conversation.query_rewriter import QueryRewriter
from app.helpers.keyword_query import fts_or_query
from app.helpers.query_normalize import canonicalize_knowledge_query
from app.kb.embedding.config import EmbeddingConfig
from app.kb.embedding.factory import create_embedding_provider
from app.kb.enums import DocumentType
from app.kb.evaluation.corpus_registry import load_corpus
from app.kb.ingestion.embedding_input import build_embedding_input, embedding_input_hash
from app.kb.ingestion.indexing import Phase12VectorStore
from app.kb.evaluation.retrieval_config import RetrievalConfig

QUERIES = [
    "What is BTE?",
    "What is a BTE hearing aid?",
    "What does BTE mean?",
    "What is Behind-The-Ear?",
    "How does a BTE work?",
    "Explain BTE hearing aids",
    "BTE hearing aid type",
]
TABLE = "chunk_embeddings_phase12"
PHASE12_VERSION = "qwen_Qwen3-Embedding-0.6B_phase12_v1"
TOP_K = 100
COMPETITORS = 8


def cosine(a, b) -> float | None:
    if a is None or b is None:
        return None
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return None
    return round(float(np.dot(va, vb) / denom), 6)


def section_label(path) -> str:
    if isinstance(path, list):
        return " / ".join(str(part) for part in path)
    return str(path or "")


def preview(text: str, limit: int = 280) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def rewrite_pair(query: str) -> dict:
    canonical = canonicalize_knowledge_query(query)
    state = ConversationState(user_message=query, product="")
    QueryRewriter().apply(state)
    rewritten = (state.query_rewritten or "").strip() or canonical or query
    return {
        "query": query,
        "canonical_query": canonical,
        "rewritten_query": rewritten,
        "rewrite_executed": bool(state.query_rewritten),
        "retrieval_query": rewritten,
    }


def index_inventory(store: Phase12VectorStore) -> dict:
    sql = text(
        f"""
        SELECT embedding_version, embedding_model, embedding_model_revision,
               kb_dataset_version, chunking_algorithm_version, embedding_input_manifest,
               embedding_dimension, COUNT(*) AS n
        FROM {TABLE}
        GROUP BY 1,2,3,4,5,6,7
        ORDER BY n DESC
        """
    )
    with store.engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(sql)]
    return {
        "table": TABLE,
        "groups": rows,
        "settings_identity": {
            "phase12_embedding_version": settings.phase12_embedding_version,
            "phase12_chunking_algorithm_version": settings.phase12_chunking_algorithm_version,
            "phase12_embedding_input_manifest": settings.phase12_embedding_input_manifest,
            "embedding_model": settings.embedding_model,
            "embedding_model_revision": settings.embedding_model_revision,
            "settings_embedding_version": settings.embedding_version,
        },
        "v2_yaml": RetrievalConfig.from_yaml(ROOT / "configs/retrieval/v2.yaml").to_dict(),
        "v3_2_yaml": RetrievalConfig.from_yaml(ROOT / "configs/retrieval/v3_2.yaml").to_dict(),
    }


def fetch_bte_rows(store: Phase12VectorStore) -> list[dict]:
    sql = text(
        f"""
        SELECT chunk_id, document_id, title, section_path, content, split_method,
               document_type, embedding_version, embedding_model, embedding_model_revision,
               kb_dataset_version, chunking_algorithm_version, embedding_input_manifest,
               embedding_input_hash, embedding_dimension, embedded_at,
               LENGTH(content) AS content_chars
        FROM {TABLE}
        WHERE embedding_version = :version
          AND (
            content ILIKE '%Behind-The-Ear%'
            OR content ILIKE '%behind the ear%'
            OR CAST(section_path AS text) ILIKE '%Behind-The-Ear%'
            OR CAST(section_path AS text) ILIKE '%BTE%'
          )
        ORDER BY document_id, chunk_id
        """
    )
    with store.engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(sql, {"version": PHASE12_VERSION})]


def fetch_embedding(store: Phase12VectorStore, chunk_id: str):
    with store._session_factory() as session:
        found = session.execute(
            select(store._model).where(
                store._model.chunk_id == chunk_id,
                store._model.embedding_version == PHASE12_VERSION,
            )
        ).scalar_one_or_none()
        if found is None:
            return None
        return list(found.embedding)


def reconstruct_input(row: dict) -> dict:
    try:
        doc_type = DocumentType(str(row.get("document_type") or "other"))
    except ValueError:
        doc_type = DocumentType.OTHER
    reconstructed = build_embedding_input(
        document_type=doc_type,
        title=str(row.get("title") or ""),
        section_path=list(row.get("section_path") or []),
        content=str(row.get("content") or ""),
    )
    stored_hash = str(row.get("embedding_input_hash") or "")
    return {
        "embedding_input": reconstructed,
        "embedding_input_hash_reconstructed": embedding_input_hash(reconstructed),
        "embedding_input_hash_stored": stored_hash,
        "hash_matches_stored": embedding_input_hash(reconstructed) == stored_hash,
    }


def pick_phase12_correct(rows: list[dict]) -> dict:
    """Best analog of catalog hearing_aid_type:bte on the persisted index."""
    scored = []
    for row in rows:
        path = section_label(row.get("section_path")).lower()
        content = (row.get("content") or "").lower()
        score = 0
        if "behind-the-ear" in path:
            score += 100
        if "behind-the-ear (bte)" in content:
            score += 40
        if "types of hearing aids" in content or "types of hearing aids" in path:
            score += 20
        if "6.3 bluup+" in path:
            score += 10
        if "radius" in path or "radius bte" in content:
            score -= 50
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else {}


def v3_2_bte_chunks() -> list[dict]:
    corpus = load_corpus("v3.2")
    hits = []
    for chunk in corpus.chunks:
        path = section_label(chunk.section_path)
        hay = f"{path} {chunk.content}".lower()
        if "behind-the-ear" not in hay and chunk.content_type != "hearing_aid_type":
            continue
        if "behind-the-ear" not in hay:
            continue
        hits.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content_type": chunk.content_type,
                "section_path": list(chunk.section_path),
                "parent_section": chunk.parent_section,
                "subsection": chunk.subsection,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "embedding_input": chunk.embedding_input,
                "embedding_input_hash": chunk.embedding_input_hash,
            }
        )
    return hits


def summarize_hit(item: dict, rank: int | None) -> dict:
    return {
        "rank": rank,
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "title": item.get("title"),
        "section": section_label(item.get("section_path")),
        "split_method": item.get("split_method"),
        "vector_score": item.get("similarity"),
        "keyword_score": item.get("keyword_score"),
        "content_preview": preview(str(item.get("content") or "")),
    }


def rank_in(results: list[dict], chunk_id: str) -> tuple[int | None, dict | None]:
    for index, item in enumerate(results, start=1):
        if item.get("chunk_id") == chunk_id:
            return index, item
    return None, None


def main() -> int:
    store = Phase12VectorStore(table_name=TABLE)
    inventory = index_inventory(store)
    bte_rows = fetch_bte_rows(store)
    for row in bte_rows:
        if hasattr(row.get("embedded_at"), "isoformat"):
            row["embedded_at"] = row["embedded_at"].isoformat()
        row["section"] = section_label(row.get("section_path"))
        recon = reconstruct_input(row)
        row["reconstructed"] = {
            "hash_matches_stored": recon["hash_matches_stored"],
            "embedding_input": recon["embedding_input"],
            "embedding_input_hash_stored": recon["embedding_input_hash_stored"],
            "embedding_input_hash_reconstructed": recon["embedding_input_hash_reconstructed"],
        }
    correct = pick_phase12_correct(bte_rows)
    v32_bte = v3_2_bte_chunks()

    config = EmbeddingConfig.from_settings()
    provider = create_embedding_provider(config)

    identity_probe = None
    if correct:
        persisted = fetch_embedding(store, correct["chunk_id"])
        reconstructed = correct["reconstructed"]["embedding_input"]
        fresh_phase12 = provider.embed_documents([reconstructed])[0]
        v32_input = v32_bte[0]["embedding_input"] if v32_bte else None
        fresh_v32 = provider.embed_documents([v32_input])[0] if v32_input else None
        identity_probe = {
            "note": "Single-text probe only. Production index was not re-embedded.",
            "phase12_correct_chunk_id": correct["chunk_id"],
            "phase12_hash_matches_stored": correct["reconstructed"]["hash_matches_stored"],
            "cosine_persisted_vs_current_model_on_reconstructed_input": cosine(persisted, fresh_phase12),
            "cosine_persisted_vs_fresh_v32_bte_input": cosine(persisted, fresh_v32) if fresh_v32 else None,
            "v3_2_bte_chunk_id": v32_bte[0]["chunk_id"] if v32_bte else None,
            "embedding_input_equal": (
                reconstructed == v32_input if v32_input else False
            ),
            "phase12_embedding_input": reconstructed,
            "v3_2_embedding_input": v32_input,
            "query_encoder": {
                "model": config.model,
                "revision": config.model_revision,
                "query_instruction": config.query_instruction,
                "normalize": config.normalize,
                "device": config.device,
            },
        }

    query_rows = []
    for query in QUERIES:
        rewrite = rewrite_pair(query)
        retrieval_query = rewrite["retrieval_query"]
        query_vector = provider.embed_queries([retrieval_query])[0]
        vector_hits = store.search(query_vector, top_k=TOP_K, embedding_version=PHASE12_VERSION)
        keyword_hits = store.search_keyword(
            retrieval_query, top_k=TOP_K, embedding_version=PHASE12_VERSION
        )
        v_rank, v_hit = rank_in(vector_hits, correct.get("chunk_id") or "")
        k_rank, k_hit = rank_in(keyword_hits, correct.get("chunk_id") or "")
        analog_ids = {row["chunk_id"] for row in bte_rows}
        analog_vector = [
            {"rank": i, **summarize_hit(item, i)}
            for i, item in enumerate(vector_hits, start=1)
            if item["chunk_id"] in analog_ids
        ][:5]
        query_rows.append(
            {
                **rewrite,
                "fts_or_query": fts_or_query(retrieval_query),
                "vector": {
                    "rank": v_rank,
                    "score": None if v_hit is None else v_hit.get("similarity"),
                    "in_top_20": v_rank is not None and v_rank <= 20,
                    "pool_size": len(vector_hits),
                    "correct": None if v_hit is None else summarize_hit(v_hit, v_rank),
                    "competitors": [summarize_hit(item, i) for i, item in enumerate(vector_hits[:COMPETITORS], start=1)],
                    "bte_analogs_in_top_100": analog_vector,
                },
                "keyword": {
                    "rank": k_rank,
                    "score": None if k_hit is None else k_hit.get("keyword_score"),
                    "in_top_20": k_rank is not None and k_rank <= 20,
                    "pool_size": len(keyword_hits),
                    "correct": None if k_hit is None else summarize_hit(k_hit, k_rank),
                    "competitors": [summarize_hit(item, i) for i, item in enumerate(keyword_hits[:COMPETITORS], start=1)],
                },
            }
        )
        if identity_probe and query == "What is BTE?":
            identity_probe["cosine_query_vs_persisted_phase12_bte"] = cosine(query_vector, fetch_embedding(store, correct["chunk_id"]))
            if v32_bte:
                identity_probe["cosine_query_vs_fresh_v32_bte"] = cosine(
                    query_vector, provider.embed_documents([v32_bte[0]["embedding_input"]])[0]
                )

    payload = {
        "production_untouched": True,
        "reindex": False,
        "index": inventory,
        "phase12_bte_rows": [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "section": row["section"],
                "split_method": row["split_method"],
                "content_chars": row["content_chars"],
                "hash_matches_stored": row["reconstructed"]["hash_matches_stored"],
                "content_preview": preview(row["content"], 400),
                "chosen_correct": row["chunk_id"] == correct.get("chunk_id"),
            }
            for row in bte_rows
        ],
        "phase12_correct_analog": {
            "chunk_id": correct.get("chunk_id"),
            "section": correct.get("section"),
            "title": correct.get("title"),
            "content": correct.get("content"),
            "embedding_input": (correct.get("reconstructed") or {}).get("embedding_input"),
            "hash_matches_stored": (correct.get("reconstructed") or {}).get("hash_matches_stored"),
            "identity": {
                "embedding_version": correct.get("embedding_version"),
                "embedding_model": correct.get("embedding_model"),
                "embedding_model_revision": correct.get("embedding_model_revision"),
                "kb_dataset_version": correct.get("kb_dataset_version"),
                "chunking_algorithm_version": correct.get("chunking_algorithm_version"),
                "embedding_input_manifest": correct.get("embedding_input_manifest"),
            },
        } if correct else None,
        "v3_2_bte_chunks": v32_bte,
        "identity_probe": identity_probe,
        "queries": query_rows,
    }

    out_dir = ROOT / "reports/mass_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bte_forensics.json"
    txt_path = out_dir / "bte_forensics.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(format_text(payload), encoding="utf-8")
    print(format_text(payload))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


def format_text(payload: dict) -> str:
    lines = [
        "BTE_CANDIDATE_RETRIEVAL_FORENSICS",
        "production_untouched: true",
        "reindex: false",
        "index: chunk_embeddings_phase12",
        "",
        "INDEX_IDENTITY",
    ]
    for group in payload["index"]["groups"]:
        lines.append(
            f"  n={group['n']} version={group['embedding_version']} model={group['embedding_model']} "
            f"rev={group['embedding_model_revision']} chunking={group['chunking_algorithm_version']} "
            f"manifest={group['embedding_input_manifest']}"
        )
    ident = payload["index"]["settings_identity"]
    lines.append(f"  settings.phase12_embedding_version={ident['phase12_embedding_version']}")
    lines.append(f"  settings.embedding_version (default/in-memory)={ident['settings_embedding_version']}")
    lines.append(f"  v2.yaml embedding_version={payload['index']['v2_yaml'].get('embedding_version')}")
    lines.append(f"  v3_2.yaml embedding_version={payload['index']['v3_2_yaml'].get('embedding_version')} mode={payload['index']['v3_2_yaml'].get('mode')}")
    lines.extend(["", "PHASE12_BTE_ANALOGS"])
    for row in payload["phase12_bte_rows"]:
        marker = " CORRECT" if row["chosen_correct"] else ""
        lines.append(
            f"  {row['chunk_id']}{marker} section={row['section']!r} chars={row['content_chars']} "
            f"hash_ok={row['hash_matches_stored']}"
        )
        lines.append(f"    {row['content_preview']}")
    correct = payload.get("phase12_correct_analog") or {}
    lines.extend(["", "PHASE12_CORRECT_EMBEDDING_INPUT"])
    lines.append(correct.get("embedding_input") or "(none)")
    lines.extend(["", "V3_2_BTE_CHUNKS (in-memory corpus, not persisted)"])
    for chunk in payload["v3_2_bte_chunks"]:
        lines.append(
            f"  {chunk['chunk_id']} type={chunk['content_type']} tokens={chunk['token_count']} "
            f"section={section_label(chunk['section_path'])!r}"
        )
        lines.append(f"    embedding_input:\n{chunk['embedding_input']}")
    probe = payload.get("identity_probe") or {}
    lines.extend(["", "IDENTITY_PROBE (not a reindex)"])
    lines.append(f"  hash_matches_stored={probe.get('phase12_hash_matches_stored')}")
    lines.append(f"  cosine(persisted, current_model(reconstructed_phase12_input))={probe.get('cosine_persisted_vs_current_model_on_reconstructed_input')}")
    lines.append(f"  cosine(persisted, current_model(v3_2_bte_input))={probe.get('cosine_persisted_vs_fresh_v32_bte_input')}")
    lines.append(f"  cosine(query 'What is BTE?', persisted phase12 analog)={probe.get('cosine_query_vs_persisted_phase12_bte')}")
    lines.append(f"  cosine(query 'What is BTE?', fresh v3.2 BTE chunk)={probe.get('cosine_query_vs_fresh_v32_bte')}")
    lines.append(f"  embedding_inputs_equal={probe.get('embedding_input_equal')}")
    lines.extend(["", "QUERIES"])
    for row in payload["queries"]:
        lines.append(f"  Q: {row['query']}")
        lines.append(f"     rewritten_query: {row['rewritten_query']} executed={row['rewrite_executed']}")
        lines.append(f"     fts: {row['fts_or_query']}")
        v, k = row["vector"], row["keyword"]
        lines.append(
            f"     vector rank={v['rank']} score={v['score']} top20={v['in_top_20']} | "
            f"keyword rank={k['rank']} score={k['score']} top20={k['in_top_20']}"
        )
        lines.append("     vector competitors:")
        for item in v["competitors"][:5]:
            lines.append(f"       {item['rank']}. {item['vector_score']} {item['section']} :: {item['content_preview']}")
        lines.append("     keyword competitors:")
        for item in k["competitors"][:5]:
            lines.append(f"       {item['rank']}. {item['keyword_score']} {item['section']} :: {item['content_preview']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
