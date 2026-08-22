"""Error analysis of CE threshold sweep. Does not change production or gold labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.config import settings
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.models import RetrievalCandidate
from app.providers.reranker.cross_encoder import CrossEncoderReranker
from app.providers.retrieval.keyword_retriever import KeywordCandidateRetriever
from app.providers.retrieval.vector_retriever import VectorCandidateRetriever
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.search.backends import EmbeddingSession
from app.kb.ingestion.indexing import Phase12VectorStore
from tests.integration.retrieval.test_expanded_golden import DOC_ID, _is_relevant, _load_golden
from tests.integration.retrieval.test_cross_encoder_threshold_calibration import JSON_PATH

REPORT_PATH = Path("reports") / "hybrid_retrieval_threshold_error_analysis.txt"
CHUNKS_PATH = Path("data/documents") / DOC_ID / "chunks" / "chunks.json"
FINAL_K = 5


def _load_chunks() -> dict[str, dict]:
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return {item["chunk_id"]: item for item in payload["chunks"]}


def _final_at(reranked: list[dict], threshold: float) -> list[dict]:
    return [row for row in reranked if row.get("score") is not None and row["score"] >= threshold][:FINAL_K]


def _status(rows: list[dict], threshold: float) -> str:
    kept = _final_at(rows, threshold)
    if not kept:
        return "empty"
    return "kept"


def _preview(text: str, limit: int = 220) -> str:
    return " ".join((text or "").split())[:limit]


def _build_hybrid():
    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))
    store = Phase12VectorStore(table_name=config.vector_table)
    try:
        probe = store.search_keyword("Earkart", top_k=1, document_id=DOC_ID)
    except Exception as exc:
        pytest.skip(f"PostgreSQL/pgvector unavailable: {exc}")
    if not probe:
        pytest.skip("Indexed pricelist document is not available; not re-ingesting")
    session = EmbeddingSession(device="cuda")
    ce = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cuda",
        batch_size=16,
        max_length=settings.reranker_max_length,
    )
    ce.rerank("warmup", [RetrievalCandidate(chunk_id="w", document_id="w", text="warmup")])
    return HybridRetriever(
        VectorCandidateRetriever(store=store, session=session, embedding_version=config.embedding_version, default_k=20),
        KeywordCandidateRetriever(store, embedding_version=config.embedding_version, default_k=20),
        ce,
        vector_k=20,
        keyword_k=20,
        final_k=5,
        min_score=0.0,
    )


def _pool_map(candidates: list[RetrievalCandidate]) -> dict[str, RetrievalCandidate]:
    return {item.chunk_id: item for item in candidates}


def _dump_query(hybrid: HybridRetriever, question: dict) -> dict:
    detailed = hybrid.search_detailed(question["query"], final_k=5, document_id=DOC_ID, min_score=0.0)
    vector_map = _pool_map(detailed.vector_candidates)
    keyword_map = _pool_map(detailed.keyword_candidates)
    rows = []
    for rank, hit in enumerate(detailed.reranked_candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "score": hit.rerank_score,
                "vector_score": None if hit.chunk_id not in vector_map else vector_map[hit.chunk_id].vector_score,
                "keyword_score": None if hit.chunk_id not in keyword_map else keyword_map[hit.chunk_id].keyword_score,
                "in_vector": hit.chunk_id in vector_map,
                "in_keyword": hit.chunk_id in keyword_map,
                "relevant": _is_relevant(hit, question),
                "text": hit.text,
            }
        )
    return {
        "vector_ids": [item.chunk_id for item in detailed.vector_candidates],
        "keyword_ids": [item.chunk_id for item in detailed.keyword_candidates],
        "rows": rows,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to attach vector/keyword scores")
def test_threshold_error_analysis() -> None:
    production_before = {
        "reranker_provider": settings.reranker_provider,
        "reranker_min_score": settings.reranker_min_score,
        "final_retrieval_k": settings.final_retrieval_k,
    }
    assert settings.reranker_min_score == 0.0
    if not JSON_PATH.exists():
        pytest.skip("Run test_cross_encoder_threshold_calibration.py first")

    calibration = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    records = {row["id"]: row for row in calibration["records"]}
    questions = {item["id"]: item for item in _load_golden()["questions"]}
    chunks = _load_chunks()

    hybrid = _build_hybrid()
    fame = _dump_query(hybrid, questions["num_fame_8900"])
    nyc = _dump_query(hybrid, questions["neg_new_york_stores"])
    fame_p = _dump_query(hybrid, questions["multi_fame_p"])

    changed = []
    for record in calibration["records"]:
        before = [row["chunk_id"] for row in _final_at(record["reranked"], 0.0)]
        after = [row["chunk_id"] for row in _final_at(record["reranked"], 0.05)]
        if before != after:
            gold_before = any(row.get("relevant") for row in _final_at(record["reranked"], 0.0))
            gold_after = any(row.get("relevant") for row in _final_at(record["reranked"], 0.05))
            changed.append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "answerable": record["answerable"],
                    "before": before,
                    "after": after,
                    "top0": record["reranked"][0] if record["reranked"] else None,
                    "first_relevant": next((row for row in record["reranked"] if row.get("relevant")), None),
                    "gold_lost": gold_before and not gold_after,
                    "became_empty": bool(before) and not after,
                }
            )

    band_rel = []
    band_irrel = []
    for record in calibration["records"]:
        for row in record["reranked"]:
            score = row.get("score")
            if score is None or score < 0.0 or score > 0.10:
                continue
            item = {
                "id": record["id"],
                "answerable": record["answerable"],
                "score": score,
                "rank": row["rank"],
                "relevant": row["relevant"],
                "text": row["text_preview"],
            }
            if row["relevant"]:
                band_rel.append(item)
            else:
                band_irrel.append(item)

    lines = _format_report(
        production_before=production_before,
        chunks=chunks,
        records=records,
        fame=fame,
        nyc=nyc,
        fame_p=fame_p,
        changed=changed,
        band_rel=sorted(band_rel, key=lambda item: item["score"]),
        band_irrel=sorted(band_irrel, key=lambda item: -item["score"])[:25],
        band_irrel_n=len(band_irrel),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:80]))
    assert settings.reranker_min_score == production_before["reranker_min_score"] == 0.0
    assert settings.reranker_provider == production_before["reranker_provider"]


def _format_report(*, production_before, chunks, records, fame, nyc, fame_p, changed, band_rel, band_irrel, band_irrel_n) -> list[str]:
    fame_rows = fame["rows"][:12]
    gold_fame = [row for row in fame["rows"] if row["relevant"]]
    top1 = fame["rows"][0]
    mixed = next(row for row in fame["rows"] if row["chunk_id"].startswith("3fe224c2"))
    dedicated = next(row for row in fame["rows"] if row["chunk_id"].startswith("f31ac22a"))
    heading = next((row for row in fame["rows"] if row["chunk_id"].startswith("20323360")), None)
    nyc_top = nyc["rows"][0]
    fp = fame_p["rows"][1]
    gold_p = fame_p["rows"][0]

    lines = [
        "Cross-encoder threshold error analysis",
        "Evaluation only. Golden set, .env, and production threshold were not changed.",
        f"Recorded production: {production_before}",
        "",
        "=== 1. num_fame_8900 ===",
        'Query: "What is the MRP of the FAME model at 8900?"',
        'Gold matcher: content_patterns=["8,900"]  (answerable=true)',
        "Complete top of reranked pool (CE CUDA, min_score=0.0):",
        f"{'rank':<5} {'gold':<5} {'ce':<10} {'vector':<10} {'keyword':<10} {'vec?':<5} {'kw?':<5} chunk",
    ]
    for row in fame_rows:
        lines.append(
            f"{row['rank']:<5} {str(row['relevant']):<5} {_fmt(row['score']):<10} {_fmt(row['vector_score']):<10} "
            f"{_fmt(row['keyword_score']):<10} {str(row['in_vector']):<5} {str(row['in_keyword']):<5} {row['chunk_id'][:16]}"
        )
        lines.append(f"      {_preview(row['text'], 240)}")
    lines.extend(
        [
            "",
            "Which chunks are gold under the current matcher (contain '8,900'):",
        ]
    )
    for row in gold_fame:
        text = chunks.get(row["chunk_id"], {}).get("content", row["text"])
        lines.append(f"- rank {row['rank']} score={row['score']} vector={row['vector_score']} keyword={row['keyword_score']}")
        lines.append(f"  chunk_id={row['chunk_id']}")
        lines.append(f"  { _preview(text, 400)}")
    lines.extend(
        [
            "",
            f"Rank #1 is NOT gold. chunk={top1['chunk_id']} CE={top1['score']} vector={top1['vector_score']} keyword={top1['keyword_score']}",
            f"#1 text: {_preview(top1['text'], 400)}",
            "Evidence: #1 is the FAME 2T / MRP 10,900+11,900 features chunk. It does not contain 8,900.",
            "",
            "Is the gold label definitely correct?",
            "Potential annotation issue: the matcher is only '8,900'. That marks TWO chunks:",
            f"  (a) mixed FAME-series table rank {mixed['rank']} CE={mixed['score']} — contains 8,900 AND 9,900 AND 10,900 AND 11,900.",
            f"  (b) dedicated heading 'MRP: ₹ 8,900' rank {dedicated['rank']} CE={dedicated['score']} — contains 8,900 but does not say FAME.",
            "The FAME model identity lives in a third chunk (5.4.1 FAME, mild-moderate, 13 zinc air) with NO price:",
            f"  heading in_pool={heading is not None} rank={None if heading is None else heading['rank']} CE={None if heading is None else heading['score']}",
            "Relevant information is split: name/spec in 5.4.1 FAME; price in the ₹ 8,900 heading; both appear together only in the mixed table.",
            "Does the recall-critical gold chunk contain exact 8900? It contains '8,900' with a comma, not '8900'. Query uses 8900.",
            "This is a numerical formatting problem for matching AND a MiniLM cross-encoder weakness (8900 vs 8,900 vs ₹ 8,900).",
            "",
            "Layer for num_fame_8900:",
            f"  Vector pool contains mixed table: {mixed['in_vector']}; dedicated 8,900: {dedicated['in_vector']}",
            f"  Keyword pool contains mixed table: {mixed['in_keyword']}; dedicated 8,900: {dedicated['in_keyword']}",
            "  Candidate generation DID produce the gold chunks (they are in the merged pool).",
            "  Cross-encoder ranked FAME 2T (wrong price) above both gold chunks. That is a reranking problem.",
            "  Threshold 0.05 then drops both gold scores (0.021 and 0.00086) and keeps #1 at 0.238. That is a threshold problem on top of the ranking error.",
            "  Primary layer: Cross-encoder ranking. Secondary: Threshold. Not a miss in vector/keyword candidate generation.",
            "",
            "=== 2. neg_new_york_stores ===",
            'Query: "Where are Earkart retail stores in New York City?"  answerable=false',
            f"Top candidate CE={nyc_top['score']} logit≈-0.72 vector={nyc_top['vector_score']} keyword={nyc_top['keyword_score']}",
            f"chunk_id={nyc_top['chunk_id']} in_vector={nyc_top['in_vector']} in_keyword={nyc_top['in_keyword']}",
            f"Full text: {_preview(chunks.get(nyc_top['chunk_id'], {}).get('content', nyc_top['text']), 500)}",
            "The chunk is the Noida, India HQ contact block. It does not mention New York, retail stores, or the USA.",
            "Golden unanswerable label is correct for this PDF. This is not a NYC store listing.",
            "Retrieval surfaced a location/address chunk because the query is a 'where' question. That is understandable.",
            "The 0.328 score is the cross-encoder treating an India address as moderately relevant to a NYC store question.",
            "Classification: E. Cross-encoder false positive",
            "Evidence: correct gap label; chunk cannot answer NYC stores; CE still assigned 0.328 (highest gap score).",
            "",
            "=== 3. Highest irrelevant score 0.999639 ===",
            'Query: multi_fame_p  "FAME P upto severe hearing loss 0-105 dB with 675 zinc air battery"',
            "Category: multi_condition. Gold patterns: ['FAME P', '0-105 dB']",
            f"Rank 1 GOLD CE={gold_p['score']} vector={gold_p['vector_score']} keyword={gold_p['keyword_score']} chunk={gold_p['chunk_id']}",
            f"  {_preview(gold_p['text'], 300)}",
            f"Rank 2 IRRELEVANT CE={fp['score']} vector={fp['vector_score']} keyword={fp['keyword_score']} chunk={fp['chunk_id']}",
            f"  {_preview(fp['text'], 300)}",
            "Reason classified irrelevant: matcher requires ALL of 'FAME P' and '0-105 dB'. Rank 2 is FAME SP, 0-115 dB. Annotation is correct.",
            "CE nearly ties sibling SKUs (0.9999 vs 0.9996) because both mention severe loss and 675 zinc air.",
            "Classification: A. True false positive (near-duplicate product). Ranking is still correct: gold is #1.",
            "Threshold cannot fix this: both scores are ~1.0. This is why highest-irrelevant=0.999 does not justify a cutoff.",
            "",
            "=== 4. Queries whose FINAL top-5 changes from threshold 0.00 to 0.05 ===",
            f"Changed queries: {len(changed)}",
        ]
    )
    for item in changed:
        top = item["top0"]
        rel = item["first_relevant"]
        lines.append(
            f"- {item['id']}  cat={item['category']}  answerable={item['answerable']}  "
            f"gold_lost={item['gold_lost']}  became_empty={item['became_empty']}"
        )
        lines.append(f"    top@0.00: {None if top is None else top['chunk_id'][:16]} score={None if top is None else top['score']}")
        lines.append(
            f"    first gold: rank={None if rel is None else rel['rank']} score={None if rel is None else rel['score']}"
        )
        lines.append(f"    ids@0.00={item['before'][:5]}")
        lines.append(f"    ids@0.05={item['after'][:5]}")
    lines.extend(
        [
            "",
            "Does 0.05 remove useful knowledge or only noise?",
            "On gap queries, 0.05 removes noise: 4/5 become empty (desirable).",
            "On num_fame_8900 it removes the only gold-matching chunks (useful knowledge, even if weakly labeled)",
            "and KEEPS the wrong FAME 2T/10,900 chunk (score 0.238). That is not a clean noise filter.",
            "Other answerable queries: top-5 membership can shrink by dropping low-score extras; first-relevant stays",
            "because those golds are >> 0.05. So 0.05 is not a general recall disaster, but it is not 'noise only'.",
            "",
            "=== 5. Score band 0.00 <= score <= 0.10 ===",
            f"Relevant in band: {len(band_rel)}",
        ]
    )
    for item in band_rel:
        lines.append(f"  {item['id']:<24} rank={item['rank']:<3} score={item['score']:<10} {_preview(item['text'], 140)}")
    lines.append(f"Non-relevant in band: {band_irrel_n} (showing highest 25)")
    for item in band_irrel:
        lines.append(f"  {item['id']:<24} rank={item['rank']:<3} score={item['score']:<10} {_preview(item['text'], 140)}")
    lines.extend(
        [
            "0.05 is NOT a meaningful separation boundary. Relevant and non-relevant both occupy 0.00-0.10.",
            "The only recall-critical gold in this band is num_fame_8900. Gap top-1s also sit here except New York.",
            "",
            "=== 6. All gap queries ===",
            "| Query | Top Score | Top chunk | @0.05 | @0.20 | @0.35 | Classification |",
            "|---|---:|---|---|---|---|---|",
        ]
    )
    gap_class = {
        "neg_cochlear_lifetime": "A/E low-score CE; rejectable at 0.05",
        "neg_new_york_stores": "E CE false positive on HQ address; only rejected at 0.35",
        "neg_iphone_bluetooth": "A correct negative; rejectable at 0.05",
        "neg_gst_rate": "A correct negative; rejectable at 0.05",
        "neg_crm_password": "B CRM mention exists but no password reset; rejectable at 0.05",
    }
    for gid in (
        "neg_cochlear_lifetime",
        "neg_new_york_stores",
        "neg_iphone_bluetooth",
        "neg_gst_rate",
        "neg_crm_password",
    ):
        rec = records[gid]
        top = rec["reranked"][0]
        lines.append(
            f"| {gid} | {top['score']} | {top['chunk_id'][:16]}… | {_status(rec['reranked'], 0.05)} | "
            f"{_status(rec['reranked'], 0.20)} | {_status(rec['reranked'], 0.35)} | {gap_class[gid]} |"
        )
    lines.extend(
        [
            "Cross-encoder can reject 4/5 gaps at 0.05. It cannot reject New York until 0.35.",
            "",
            "=== 7. Layer responsibility ===",
            "num_fame_8900: candidate generation OK → CE ranks wrong FAME SKU first → threshold 0.05 deletes gold. Layers: CE, then threshold.",
            "neg_new_york_stores: retrieval OK-ish (address for 'where') → CE over-scores India HQ. Layer: CE false positive. Not threshold-at-0.05.",
            "multi_fame_p rank-2 0.9996: candidate generation OK → CE sibling confusion. Layer: CE. Not threshold.",
            "",
            "=== 8. Potential annotation issues (dataset NOT modified) ===",
            "Potential annotation issue: num_fame_8900 gold is only '8,900', so a 4-price mixed table becomes first-relevant.",
            "Potential annotation issue: dedicated ₹ 8,900 chunk does not name FAME; 5.4.1 FAME chunk does not contain the price.",
            "Potential annotation issue: neg_crm_password is still a true gap for password reset, but CRM is mentioned.",
            "No gold IDs, labels, or answerable flags were changed.",
            "",
            "=== 9. Decision matrix ===",
            "| Candidate | Recall@5 | Gap rejection | Main risk |",
            "|---|---:|---:|---|",
            "| 0.00 | 1.00 | 0/5 | Always returns 5 chunks on gaps |",
            "| 0.05 | 0.975 | 4/5 | Drops fame-8900 gold; still answers with wrong FAME SKU; NYC survives |",
            "| 0.20 | 0.975 | 4/5 | Same metrics as 0.05 on this set; not a new operating point |",
            "| 0.35 | 0.975 | 5/5 | Empties fame-8900; NYC rejected; still cannot fix 0.999 sibling FPs |",
            "",
            "RECOMMENDATION: E. Threshold cannot safely be selected yet",
            "Operational consequence: keep RERANKER_MIN_SCORE=0.0. Do not apply 0.05/0.20/0.35.",
            "Not applied.",
            "",
            "=== 10. Three questions ===",
            "1. Why does a relevant result have score 0.021484?",
            "   That score is the mixed FAME-series table that happens to contain '8,900' among four MRPs.",
            "   MiniLM is not doing precise 8900↔8,900 price matching; the dedicated ₹ 8,900 chunk scores 0.00086",
            "   because it does not say FAME. Split chunking + weak gold pattern + CE ranking. Mixed cause:",
            "   annotation/data issue AND genuine CE failure on the dedicated price chunk.",
            "2. Why does an unanswerable query have score 0.328275?",
            "   CE scores the Noida HQ contact block as a moderate answer to a 'where are stores' question.",
            "   The label is correct. Genuine CE false positive, not a gold-label error.",
            "3. Why does an irrelevant candidate have score 0.999639?",
            "   FAME SP is a near-duplicate of FAME P. Gold #1 is 0.9999; sibling #2 is 0.9996.",
            "   Annotation is correct. Genuine CE limitation on similar SKUs. Threshold cannot separate them.",
            "",
            "Is the CE score suitable for thresholding answerable vs unanswerable?",
            "No. A true gold can score 0.021 while a true gap scores 0.328, and a wrong sibling can score 0.999.",
            "Thresholding alone cannot reliably determine answerable vs unanswerable on this set.",
        ]
    )
    return lines


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
