"""Tests for version-agnostic retrieval/evaluation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.kb.enums import DocumentType, ExtractionMethod
from app.kb.evaluation.comparison_runner import compare_retrieval
from app.kb.evaluation.models import EvaluationQuestion, ExpectedAnchor, RankedHit
from app.kb.evaluation.report.retrieval_report import report_paths
from app.kb.evaluation.resolver.expected import load_evaluation_questions, resolve_anchor
from app.kb.evaluation.retrieval_config import RetrievalConfig
from app.kb.evaluation.retrieval_runner import evaluate_question, run_retrieval_evaluation
from app.kb.evaluation.search.backends import RetrievalBackend
from app.kb.ingestion.models import DocumentRecord, DocumentUploadStatus, Phase12ChunkRecord
from app.kb.ingestion.phase16_corpus import Phase16CorpusDocument
from app.kb.ingestion.phase16_corpus_v3_1 import SUMMARY_CONTENT_TYPE, build_why_choose_summary_chunk
from app.kb.retrieval.service import RetrievalService


def _chunk(
    chunk_id: str,
    *,
    content: str,
    document_id: str = "doc-a",
    section_path: list[str] | None = None,
    content_type: str = "paragraph",
) -> Phase12ChunkRecord:
    return Phase12ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version=1,
        document_type=DocumentType.COMPANY,
        chunking_algorithm_version="test",
        content=content,
        embedding_input=content,
        embedding_input_hash="hash",
        token_count=len(content.split()),
        section_path=section_path or ["Section"],
        content_type=content_type,
        source_file_hash="file",
        extraction_method=ExtractionMethod.NATIVE,
        created_at=datetime.now(timezone.utc),
    )


class _StaticBackend(RetrievalBackend):
    def __init__(self, hits_by_query: dict[str, list[str]]) -> None:
        self._hits = hits_by_query
        self._chunks: dict[str, Phase12ChunkRecord] = {}

    def register_chunk(self, chunk: Phase12ChunkRecord) -> None:
        self._chunks[chunk.chunk_id] = chunk

    def search(self, query: str, *, top_k: int) -> list[RankedHit]:
        ids = self._hits.get(query, [])[:top_k]
        results: list[RankedHit] = []
        for index, chunk_id in enumerate(ids, start=1):
            chunk = self._chunks.get(chunk_id) or _chunk(chunk_id, content=f"body {chunk_id}")
            results.append(
                RankedHit(
                    rank=index,
                    similarity=1.0 - index * 0.01,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    section_path=list(chunk.section_path),
                    token_count=chunk.token_count,
                    content_type=chunk.content_type,
                    text=chunk.content,
                )
            )
        return results


def test_retrieval_config_from_yaml() -> None:
    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v2.yaml"))
    assert config.corpus_version == "v2"
    assert config.vector_table == "chunk_embeddings_v3_2"
    assert config.mode == "pgvector"


def test_retrieval_config_v3_1_uses_in_memory() -> None:
    config = RetrievalConfig.from_yaml(Path("configs/retrieval/v3_1.yaml"))
    assert config.corpus_version == "v3.1"
    assert config.mode == "in-memory"
    assert config.chunking_algorithm_version == "phase16.1.0"


def test_stable_anchor_resolves_delivery_without_faq_false_positive() -> None:
    policy = _chunk(
        "policy-delivery",
        content="Availability of products and delivery timelines may vary by location.",
        document_id="terms-doc",
        section_path=["Terms", "4.5 Delivery Timeline Variance"],
    )
    faq = _chunk(
        "faq-deliver",
        content="The hearing aid delivers the amplified sound to the ear canal.",
        document_id="merged-doc",
        section_path=["Merged", "6.3 Bluup+"],
    )
    anchor = ExpectedAnchor(
        knowledge_key="policy:delivery_timeline",
        document_id="terms-doc",
        section_path_contains=["4.5 Delivery Timeline Variance"],
        content_patterns=["delivery timeline"],
        exclude_patterns=["delivers the amplified sound"],
    )
    resolved = resolve_anchor(anchor, [policy, faq])
    assert resolved.expected_chunk_ids == ["policy-delivery"]
    assert resolved.answerability == "ANSWERABLE"


def test_why_choose_prefers_summary_chunk_in_v3_1() -> None:
    atom = _chunk(
        "atom-benefit",
        content="Free Batteries / Dehumidifier — Free UV dehumidifier",
        section_path=["Merged", "5. Why Choose earKART"],
        content_type="product_feature",
    )
    summary = _chunk(
        "summary-benefit",
        content="Why Choose earKART\nFree Insurance for Your Hearing Aid",
        section_path=["Merged", "5. Why Choose earKART"],
        content_type=SUMMARY_CONTENT_TYPE,
    )
    anchor = ExpectedAnchor(
        knowledge_key="benefits:why_choose_summary",
        section_path_contains=["Why Choose earKART"],
        content_patterns=["why choose earkart"],
    )
    resolved = resolve_anchor(anchor, [atom, summary])
    assert resolved.expected_chunk_ids == ["summary-benefit"]


def test_evaluate_question_metrics() -> None:
    expected = resolve_anchor(
        ExpectedAnchor(knowledge_key="test:key", content_patterns=["target"]),
        [_chunk("target", content="target content")],
    )
    backend = _StaticBackend({"q": ["other", "target"]})
    backend.register_chunk(_chunk("target", content="target content"))
    backend.register_chunk(_chunk("other", content="other content"))
    question = EvaluationQuestion(id="T1", question="q", anchor=ExpectedAnchor(knowledge_key="test:key"))
    result = evaluate_question(question, backend, resolved_expected=expected, top_k=10)
    assert result.recall_at_10 == 1.0
    assert result.expected_chunk_rank == 2
    assert result.passed_at_10 is True
    assert result.root_cause is None


def test_run_retrieval_evaluation_with_mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        _chunk("c1", content="behind-the-ear (bte) hearing aid"),
        _chunk("c2", content="unrelated"),
    ]
    backend = _StaticBackend({"what is this bte?": ["c1", "c2"]})
    for chunk in chunks:
        backend.register_chunk(chunk)

    config = RetrievalConfig.from_dict(
        {
            "corpus_version": "v-test",
            "vector_table": None,
            "embedding_model": "test",
            "embedding_model_revision": "rev",
            "embedding_version": "v1",
            "kb_dataset_version": "kb",
            "chunking_algorithm_version": "algo",
            "embedding_input_manifest": "manifest",
            "mode": "in-memory",
            "top_k": 10,
        }
    )
    questions = [
        EvaluationQuestion(
            id="ERK-V2-001",
            question="what is this bte?",
            anchor=ExpectedAnchor(
                knowledge_key="hearing_aid_type:bte",
                content_patterns=["behind-the-ear (bte)", "bte"],
            ),
        )
    ]

    monkeypatch.setattr(
        "app.kb.evaluation.retrieval_runner.load_corpus",
        lambda _version: type("Corpus", (), {"chunks": chunks, "source_pdfs": [], "metadata": {}})(),
    )
    monkeypatch.setattr(
        "app.kb.evaluation.retrieval_runner.build_backend",
        lambda *_args, **_kwargs: backend,
    )

    report = run_retrieval_evaluation(config=config, questions=questions)
    assert report["metrics"]["recall_at_1"] == 1.0
    assert report["queries"][0]["expected_chunk_rank"] == 1


def test_compare_retrieval_reports_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*, config, questions, device=None, top_n_report=10):
        rank = 5 if config.corpus_version == "v3" else 2
        return {
            "metrics": {
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 1.0 if rank <= 10 else 0.0,
                "mrr": 1.0 / rank,
                "passed_at_1": 0,
                "passed_at_3": 0,
                "passed_at_5": 0,
                "passed_at_10": 1 if rank <= 10 else 0,
                "total_questions": len(questions),
                "answerable_questions": len(questions),
            },
            "queries": [
                {
                    "id": question.id,
                    "expected_chunk_rank": rank,
                    "passed_at_10": rank <= 10,
                    "root_cause": None if rank <= 10 else "UNRESOLVED",
                }
                for question in questions
            ],
        }

    monkeypatch.setattr("app.kb.evaluation.comparison_runner.run_retrieval_evaluation", _fake_run)
    baseline = RetrievalConfig.from_dict(
        {
            "corpus_version": "v3",
            "vector_table": None,
            "embedding_model": "m",
            "embedding_model_revision": "r",
            "embedding_version": "v",
            "kb_dataset_version": "kb",
            "chunking_algorithm_version": "a",
            "embedding_input_manifest": "m",
            "mode": "in-memory",
        }
    )
    candidate = RetrievalConfig.from_dict({**baseline.to_dict(), "corpus_version": "v3.1"})
    comparison = compare_retrieval(
        baseline_config=baseline,
        candidate_config=candidate,
        questions_path=Path("data/evaluations/earkart_kb_v2_9.json"),
    )
    assert comparison["metric_deltas"]["mrr"] > 0
    assert comparison["per_query"][0]["improved"] is True


def test_report_paths_use_dataset_and_version() -> None:
    json_path, txt_path = report_paths(Path("reports"), "earkart_kb_v2_9", "v3.1")
    assert json_path.name == "earkart_kb_v2_9_v3_1.json"
    assert txt_path.name == "earkart_kb_v2_9_v3_1.txt"


def test_dataset_json_loads_nine_questions() -> None:
    questions = load_evaluation_questions(Path("data/evaluations/earkart_kb_v2_9.json"))
    assert len(questions) == 9
    delivery = next(item for item in questions if item.id == "ERK-V2-005")
    assert delivery.anchor.knowledge_key == "policy:delivery_timeline"
    assert "delivers the amplified sound" in delivery.anchor.exclude_patterns


def test_retrieval_service_delegates_to_backend() -> None:
    backend = _StaticBackend({"hello": ["c1"]})
    backend.register_chunk(_chunk("c1", content="hello world"))
    service = RetrievalService(backend)
    hits = service.search("hello", top_k=5)
    assert hits[0].chunk_id == "c1"


def test_v3_1_summary_chunk_builder() -> None:
    record = DocumentRecord(
        document_id="merged-id",
        filename="merged.pdf",
        mime_type="application/pdf",
        file_hash="abc",
        document_type=DocumentType.COMPANY,
        created_at=datetime.now(timezone.utc),
        status=DocumentUploadStatus.VALIDATED,
    )
    atoms = [
        _chunk(
            "a1",
            content="Free Insurance for Your Hearing Aid",
            section_path=["Merged", "5. Why Choose earKART"],
            content_type="product_feature",
        ),
        _chunk(
            "a2",
            content="Free Batteries / Dehumidifier",
            section_path=["Merged", "5. Why Choose earKART"],
            content_type="product_feature",
        ),
    ]
    doc = Phase16CorpusDocument(
        label="merged",
        filename="merged.pdf",
        record=record,
        title="Merged",
        chunks=atoms,
    )
    summary = build_why_choose_summary_chunk(merged_doc=doc, atoms=atoms, chunk_index=2)
    assert summary is not None
    assert summary.content_type == SUMMARY_CONTENT_TYPE
    assert "Why Choose earKART" in summary.content
    assert summary.chunking_algorithm_version == "phase16.1.0"


def test_changing_version_does_not_change_eval_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_load(version: str):
        calls.append(version)
        return type("Corpus", (), {"chunks": [], "source_pdfs": [], "metadata": {}})()

    monkeypatch.setattr("app.kb.evaluation.retrieval_runner.load_corpus", _fake_load)
    monkeypatch.setattr(
        "app.kb.evaluation.retrieval_runner.build_backend",
        lambda *_a, **_k: _StaticBackend({}),
    )

    questions = load_evaluation_questions(Path("data/evaluations/earkart_kb_v2_9.json"))
    for version in ("v3", "v3.1"):
        config = RetrievalConfig.from_yaml(Path(f"configs/retrieval/{version.replace('.', '_')}.yaml"))
        run_retrieval_evaluation(config=config, questions=questions)
    assert calls == ["v3", "v3.1"]
