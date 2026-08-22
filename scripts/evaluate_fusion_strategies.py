#!/usr/bin/env python3
"""Compare eval-only fusion strategies. Does not change production retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.catalog import items_by_key, load_knowledge_catalog
from app.evaluation.fusion import FusionConfig, StrategyName
from app.evaluation.fusion_eval import compare_lists
from app.kb.evaluation.resolver.expected import load_evaluation_questions

CASES = [
    {"id": "GEN-000005", "turn": 1, "query": "What is this bte?", "key": "hearing_aid_type:bte", "family": "bte"},
    {"id": "GEN-000018", "turn": 1, "query": "What is BTE?", "key": "hearing_aid_type:bte", "family": "bte"},
    {"id": "GEN-000029", "turn": 1, "query": "what is this bte?", "key": "hearing_aid_type:bte", "family": "bte"},
    {"id": "GEN-000041", "turn": 3, "query": "What is this bte?", "key": "hearing_aid_type:bte", "family": "bte"},
    {"id": "GEN-000007", "turn": 3, "query": "Can you explain BTE work?", "key": "faq:how_hearing_aid_works", "family": "bte"},
    {"id": "GEN-000039", "turn": 1, "query": "What is BTE work?", "key": "faq:how_hearing_aid_works", "family": "bte"},
    {"id": "GEN-000070", "turn": 1, "query": "Give me a quick overview of Bluup..", "key": "product:bluup_overview", "family": "bluup"},
    {"id": "GEN-000044", "turn": 1, "query": "What's the deal with benefits of buying from Earkart?", "key": "benefits:why_choose_summary", "family": "benefits"},
    {"id": "GEN-000053", "turn": 3, "query": "What is benefits of buying from Earkart?", "key": "benefits:why_choose_summary", "family": "benefits"},
    {"id": "GEN-000065", "turn": 3, "query": "like why to buy from u?", "key": "benefits:why_choose_summary", "family": "benefits"},
    {"id": "GEN-000011", "turn": 3, "query": "Give me a quick overview of office of earkart.", "key": "company:office_location", "family": "office"},
]

STRATEGIES: list[StrategyName] = ["current_merge", "rrf", "weighted", "rank_normalized"]
BASELINE_44 = {"recall_at_10": 0.8947, "mrr": 0.6964, "passed_at_10": 34, "answerable": 38}


def load_fusion_config(path: Path) -> tuple[FusionConfig, list[StrategyName]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = list(data.get("strategies") or STRATEGIES)
    return FusionConfig.from_dict(data), names


def build_hybrid():
    from app.providers.retrieval.factory import build_knowledge_hybrid_retriever
    from app.kb.evaluation.retrieval_config import RetrievalConfig
    from app.kb.retrieval.service import RetrievalService

    config = RetrievalConfig.from_yaml(ROOT / "configs" / "retrieval" / "v2.yaml")
    vector_service = RetrievalService.from_config(config)
    return build_knowledge_hybrid_retriever(config, vector_service=vector_service), config


def retrieve_pools(hybrid, query: str):
    """Vector/keyword candidate lists only. Does not call production merge or rerank."""
    vector = hybrid._vector_retriever.retrieve(query, top_k=hybrid._vector_k)
    keyword = hybrid._keyword_retriever.retrieve(query, top_k=hybrid._keyword_k)
    return vector, keyword


def summarize_hits(rows: list[dict], strategies: list[str]) -> dict:
    payload = {name: {"hit_at_10": 0, "mrr": 0.0} for name in ["vector", "keyword", *strategies]}
    n = len(rows) or 1
    for row in rows:
        if row["vector"]["hit_at_10"]:
            payload["vector"]["hit_at_10"] += 1
        payload["vector"]["mrr"] += row["vector"]["mrr"]
        if row["keyword"]["hit_at_10"]:
            payload["keyword"]["hit_at_10"] += 1
        payload["keyword"]["mrr"] += row["keyword"]["mrr"]
        for name in strategies:
            payload[name]["hit_at_10"] += int(row["strategies"][name]["hit_at_10"])
            payload[name]["mrr"] += row["strategies"][name]["mrr"]
    for name in payload:
        payload[name]["hit_at_10"] = payload[name]["hit_at_10"]
        payload[name]["hit_rate"] = round(payload[name]["hit_at_10"] / n, 4)
        payload[name]["mrr"] = round(payload[name]["mrr"] / n, 4)
    return payload


def run_eleven(hybrid, catalog, config: FusionConfig, strategies: list[StrategyName]) -> dict:
    cache: dict[str, tuple] = {}
    rows = []
    for case in CASES:
        query = case["query"]
        if query not in cache:
            cache[query] = retrieve_pools(hybrid, query)
        vector, keyword = cache[query]
        item = catalog[case["key"]]
        compared = compare_lists(vector, keyword, item, config, strategies)
        rows.append(
            {
                "conversation_id": case["id"],
                "turn": case["turn"],
                "query": query,
                "key": case["key"],
                "family": case["family"],
                "catalog_question": item.question,
                **compared,
            }
        )
    bte = [row for row in rows if row["family"] == "bte"]
    locus = {}
    for row in bte:
        locus[row["failure_locus"]] = locus.get(row["failure_locus"], 0) + 1
    return {
        "cases": rows,
        "summary": summarize_hits(rows, strategies),
        "bte": {
            "count": len(bte),
            "locus_counts": locus,
            "queries": [
                {
                    "id": row["conversation_id"],
                    "query": row["query"],
                    "vector_pool_rank": row["vector"]["pool_rank"],
                    "vector_hit_at_10": row["vector"]["hit_at_10"],
                    "vector_matched_section": row["vector"]["matched_section"],
                    "keyword_pool_rank": row["keyword"]["pool_rank"],
                    "keyword_matched_section": row["keyword"]["matched_section"],
                    "current_merge_rank": row["strategies"]["current_merge"]["rank"],
                    "rrf_rank": row["strategies"]["rrf"]["rank"],
                    "weighted_rank": row["strategies"]["weighted"]["rank"],
                    "rank_normalized_rank": row["strategies"]["rank_normalized"]["rank"],
                    "locus": row["failure_locus"],
                }
                for row in bte
            ],
        },
    }


def run_hybrid_catalog(hybrid, catalog, questions, config: FusionConfig, strategies: list[StrategyName]) -> dict:
    rows = []
    for question in questions:
        key = question.anchor.knowledge_key
        item = catalog.get(key)
        if item is None or not item.answerable:
            continue
        vector, keyword = retrieve_pools(hybrid, question.question)
        compared = compare_lists(vector, keyword, item, config, strategies)
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "key": key,
                **compared,
            }
        )
    return {"metrics": summarize_hits(rows, strategies), "n": len(rows), "questions": rows}


def run_official_44(device: str | None) -> dict:
    from app.kb.evaluation.retrieval_config import RetrievalConfig
    from app.kb.evaluation.retrieval_runner import run_retrieval_evaluation

    config = RetrievalConfig.from_yaml(ROOT / "configs" / "retrieval" / "v3_2.yaml")
    questions = load_evaluation_questions(ROOT / "data" / "evaluations" / "earkart_kb_v3_1_comprehensive.json")
    report = run_retrieval_evaluation(config=config, questions=questions, device=device, top_n_report=10)
    metrics = report["metrics"]
    return {
        "metrics": metrics,
        "baseline": BASELINE_44,
        "delta_recall_at_10": round(metrics["recall_at_10"] - BASELINE_44["recall_at_10"], 4),
        "delta_mrr": round(metrics["mrr"] - BASELINE_44["mrr"], 4),
        "regressed": metrics["recall_at_10"] < BASELINE_44["recall_at_10"] - 0.02,
        "root_causes": report.get("root_causes") or report.get("root_cause_summary"),
        "questions": len(report.get("results") or questions),
    }


def format_text(payload: dict) -> str:
    lines = [
        "FUSION_STRATEGY_COMPARISON",
        "production_untouched: true",
        "",
        "ELEVEN_GENUINE_CASES",
    ]
    eleven = payload.get("eleven")
    if not eleven:
        lines.append("  (not run)")
    else:
        summary = eleven["summary"]
        for name, row in summary.items():
            lines.append(
                f"  {name}: hit@10={row['hit_at_10']}/{len(eleven['cases'])} rate={row['hit_rate']} mrr={row['mrr']}"
            )
    if eleven:
        lines.extend(["", "BTE_LOCUS"])
        bte = eleven["bte"]
        lines.append(f"  counts: {bte['locus_counts']}")
        for row in bte["queries"]:
            lines.append(
                f"  {row['id']}: vector={row['vector_pool_rank']} keyword={row['keyword_pool_rank']} "
                f"current={row['current_merge_rank']} rrf={row['rrf_rank']} weighted={row['weighted_rank']} "
                f"rank_norm={row['rank_normalized_rank']} locus={row['locus']}"
            )
    if payload.get("official_44"):
        official = payload["official_44"]
        metrics = official["metrics"]
        lines.extend(
            [
                "",
                "V3_2_44_QUESTION_REGRESSION (in-memory official)",
                f"  recall@10={metrics.get('recall_at_10')} baseline={official['baseline']['recall_at_10']} delta={official['delta_recall_at_10']}",
                f"  mrr={metrics.get('mrr')} baseline={official['baseline']['mrr']} delta={official['delta_mrr']}",
                f"  passed@10={metrics.get('passed_at_10')} baseline={official['baseline']['passed_at_10']}",
                f"  regressed={official['regressed']}",
            ]
        )
    if payload.get("hybrid_44"):
        lines.extend(["", "LIVE_HYBRID_44 (pgvector phase12, fusion overlay)"])
        for name, row in payload["hybrid_44"]["metrics"].items():
            n = payload["hybrid_44"]["n"]
            lines.append(f"  {name}: hit@10={row['hit_at_10']}/{n} rate={row['hit_rate']} mrr={row['mrr']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/fusion.yaml")
    parser.add_argument("--skip-official-44", action="store_true")
    parser.add_argument("--skip-hybrid-44", action="store_true")
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Run in-memory V3.2 44-q only; merge into an existing fusion_comparison.json",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="reports/mass_eval")
    args = parser.parse_args()
    fusion_config, strategies = load_fusion_config(ROOT / args.config)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fusion_comparison.json"
    txt_path = output_dir / "fusion_comparison.txt"

    if args.official_only:
        payload = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {
            "fusion_config": fusion_config.__dict__,
            "strategies": strategies,
        }
        payload["official_44"] = run_official_44(args.device)
    else:
        catalog = items_by_key(load_knowledge_catalog(ROOT / "data/evaluations/earkart_kb_v3_1_comprehensive.json"))
        hybrid, _retrieval_config = build_hybrid()
        payload = {
            "fusion_config": fusion_config.__dict__,
            "strategies": strategies,
            "eleven": run_eleven(hybrid, catalog, fusion_config, strategies),
        }
        if not args.skip_hybrid_44:
            questions = load_evaluation_questions(ROOT / "data/evaluations/earkart_kb_v3_1_comprehensive.json")
            payload["hybrid_44"] = run_hybrid_catalog(hybrid, catalog, questions, fusion_config, strategies)
        if not args.skip_official_44:
            payload["official_44"] = run_official_44(args.device)

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(format_text(payload), encoding="utf-8")
    print(format_text(payload))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
