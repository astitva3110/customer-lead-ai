"""Version-agnostic retrieval evaluation engine."""

from __future__ import annotations

from typing import Any

from app.kb.evaluation.corpus_registry import load_corpus
from app.kb.evaluation.metrics import recall_at_k, reciprocal_rank
from app.kb.evaluation.models import EvaluationQuestion, QueryEvaluationResult, RankedHit
from app.kb.evaluation.resolver.expected import build_document_label_map, resolve_question_expected
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.root_cause.classifier import classify_root_cause
from app.kb.evaluation.search.backends import RetrievalBackend, build_backend


def _rank_for_ids(hits: list[RankedHit], chunk_ids: list[str]) -> int | None:
    id_set = set(chunk_ids)
    for hit in hits:
        if hit.chunk_id in id_set:
            return hit.rank
    return None


def _best_document_rank(hits: list[RankedHit], document_ids: list[str]) -> int | None:
    doc_set = set(document_ids)
    best: int | None = None
    for hit in hits:
        if hit.document_id in doc_set:
            best = hit.rank if best is None else min(best, hit.rank)
    return best


def evaluate_question(
    question: EvaluationQuestion,
    backend: RetrievalBackend,
    *,
    resolved_expected,
    top_k: int,
    top_n_report: int = 10,
) -> QueryEvaluationResult:
    hits = backend.search(question.question, top_k=top_k)
    retrieved_ids = [hit.chunk_id for hit in hits]
    expected_ids = resolved_expected.expected_chunk_ids

    recall1 = recall_at_k(expected_ids, retrieved_ids, 1) if expected_ids else 0.0
    recall3 = recall_at_k(expected_ids, retrieved_ids, 3) if expected_ids else 0.0
    recall5 = recall_at_k(expected_ids, retrieved_ids, 5) if expected_ids else 0.0
    recall10 = recall_at_k(expected_ids, retrieved_ids, 10) if expected_ids else 0.0
    mrr = reciprocal_rank(expected_ids, retrieved_ids) if expected_ids else 0.0

    chunk_rank = _rank_for_ids(hits, expected_ids)
    doc_rank = _best_document_rank(hits, resolved_expected.expected_document_ids)
    passed_at_1 = recall1 == 1.0
    passed_at_3 = recall3 == 1.0
    passed_at_5 = recall5 == 1.0
    passed_at_10 = recall10 == 1.0

    root_cause = classify_root_cause(
        answerability=resolved_expected.answerability,
        expected_chunk_ids=expected_ids,
        expected_chunk_rank=chunk_rank,
        expected_document_best_rank=doc_rank,
        passed_at_10=passed_at_10,
        top_results=[hit.to_dict() for hit in hits],
        knowledge_key=resolved_expected.knowledge_key,
        expected_chunk_count=len(expected_ids),
    )

    failure_analysis = None
    if not passed_at_10 and resolved_expected.answerability == "ANSWERABLE":
        expected_set = set(expected_ids)
        expected_docs = set(resolved_expected.expected_document_ids)
        failure_analysis = {
            "top_competing_chunks": [
                hit.to_dict() for hit in hits if hit.chunk_id not in expected_set
            ][:10],
            "top_competing_documents": sorted(
                {hit.document_id for hit in hits[:20] if hit.document_id not in expected_docs}
            ),
        }

    return QueryEvaluationResult(
        id=question.id,
        question=question.question,
        answerability=resolved_expected.answerability,
        expected=resolved_expected,
        expected_chunk_rank=chunk_rank,
        expected_document_best_rank=doc_rank,
        recall_at_1=recall1,
        recall_at_3=recall3,
        recall_at_5=recall5,
        recall_at_10=recall10,
        mrr=mrr,
        passed_at_1=passed_at_1,
        passed_at_3=passed_at_3,
        passed_at_5=passed_at_5,
        passed_at_10=passed_at_10,
        root_cause=root_cause,
        top_results=hits[:top_n_report],
        retrieved_chunk_ids=retrieved_ids,
        category=question.category,
        expected_answer_type=question.expected_answer_type,
        failure_analysis=failure_analysis,
    )


def aggregate_eval_metrics(results: list[QueryEvaluationResult]) -> dict[str, Any]:
    answerable = [item for item in results if item.answerability != "CORPUS_GAP"]
    pool = answerable or results
    total = len(pool)
    if total == 0:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "passed_at_1": 0,
            "passed_at_3": 0,
            "passed_at_5": 0,
            "passed_at_10": 0,
            "total_questions": len(results),
            "answerable_questions": 0,
        }
    return {
        "recall_at_1": round(sum(item.recall_at_1 for item in pool) / total, 4),
        "recall_at_3": round(sum(item.recall_at_3 for item in pool) / total, 4),
        "recall_at_5": round(sum(item.recall_at_5 for item in pool) / total, 4),
        "recall_at_10": round(sum(item.recall_at_10 for item in pool) / total, 4),
        "mrr": round(sum(item.mrr for item in pool) / total, 4),
        "passed_at_1": sum(1 for item in answerable if item.passed_at_1),
        "passed_at_3": sum(1 for item in answerable if item.passed_at_3),
        "passed_at_5": sum(1 for item in answerable if item.passed_at_5),
        "passed_at_10": sum(1 for item in answerable if item.passed_at_10),
        "total_questions": len(results),
        "answerable_questions": len(answerable),
    }


def summarize_root_causes(results: list[QueryEvaluationResult]) -> dict[str, int]:
    summary = {label: 0 for label in [
        "CORPUS_GAP",
        "EXPECTED_DOCUMENT_WRONG_CHUNK",
        "RETRIEVAL_COMPETITION",
        "QUERY_MISMATCH",
        "EMBEDDING_REPRESENTATION",
        "RETRIEVAL_CONFIGURATION",
        "EVALUATION_MAPPING",
        "CHUNK_GRANULARITY",
        "CHUNK_FRAGMENTATION",
        "SUMMARY_MISSING",
        "UNRESOLVED",
    ]}
    for item in results:
        if item.root_cause and item.root_cause in summary:
            summary[item.root_cause] += 1
    return summary


def run_retrieval_evaluation(
    *,
    config: RetrievalConfig,
    questions: list[EvaluationQuestion],
    device: str | None = None,
    top_n_report: int = 10,
) -> dict[str, Any]:
    corpus = load_corpus(config.corpus_version)
    label_map = build_document_label_map(corpus.chunks, corpus.source_pdfs)
    backend = build_backend(config, chunks=corpus.chunks, device=device)

    results: list[QueryEvaluationResult] = []
    for question in questions:
        resolved = resolve_question_expected(question, corpus.chunks, document_labels=label_map)
        results.append(
            evaluate_question(
                question,
                backend,
                resolved_expected=resolved,
                top_k=config.top_k,
                top_n_report=top_n_report,
            )
        )

    metrics = aggregate_eval_metrics(results)
    return {
        "dataset": config.corpus_version,
        "corpus_version": config.corpus_version,
        "vector_table": config.vector_table,
        "mode": config.mode,
        "embedding_model": config.embedding_model,
        "embedding_model_revision": config.embedding_revision,
        "embedding_version": config.embedding_version,
        "kb_dataset_version": config.kb_dataset_version,
        "chunking_algorithm_version": config.chunking_algorithm_version,
        "embedding_input_manifest": config.embedding_input_manifest,
        "total_questions": len(results),
        "metrics": metrics,
        "root_cause_summary": summarize_root_causes(results),
        "queries": [item.to_dict() for item in results],
        "corpus_metadata": corpus.metadata,
    }
