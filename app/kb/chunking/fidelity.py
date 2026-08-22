"""Regression fidelity checks for chunking dry-run."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.kb.chunking.models import ChunkRecord
from app.kb.chunking.semantic_units import effective_document_type

RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"
RADIUS_M16_DOC_ID = "5f2ec8ed-c57d-5511-8d23-df8adf1b548d"
PROSPECTUS_DOC_IDS = (
    "01e76cb2-317f-509a-b8b5-f473f5bd5400",
    "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9",
)

PRODUCT_SPEC_CHECKS = (
    "Battery Life : 270 Hrs",
    "Battery Size (Zinc Air) 13",
    "Attack Time 28 ms",
    "Release Time 891 ms",
)

POLICY_CLAUSE_CHECKS = (
    "1. Product-Specific Return & Replacement Windows",
    "1.1.1 Return/Replacement Period: Within ten (10) calendar days",
    "1.1.2 Eligibility:",
    "1.2.1 Return/Replacement Period: Within seven (7) calendar days",
    "1.2.2 Eligibility:",
    "2. General Conditions",
    "3. Process",
    "4. Refunds",
    "5. Statutory Rights",
)


@dataclass
class FidelityResult:
    name: str
    passed: bool
    evidence: list[str] = field(default_factory=list)


def _chunks_for_document(chunks: list[ChunkRecord], document_id: str) -> list[ChunkRecord]:
    return [chunk for chunk in chunks if chunk.document_id == document_id]


def check_product_spec_fidelity(chunks: list[ChunkRecord]) -> FidelityResult:
    doc_chunks = _chunks_for_document(chunks, RADIUS_M16_DOC_ID)
    combined = "\n".join(chunk.content for chunk in doc_chunks)
    evidence: list[str] = []
    passed = True
    for needle in PRODUCT_SPEC_CHECKS:
        ok = needle in combined
        evidence.append(f"{needle}: {'PASS' if ok else 'FAIL'}")
        passed = passed and ok
    return FidelityResult(name="product_spec_fidelity", passed=passed, evidence=evidence)


def check_policy_fidelity(chunks: list[ChunkRecord]) -> FidelityResult:
    doc_chunks = _chunks_for_document(chunks, RETURNS_DOC_ID)
    combined = "\n".join(chunk.content for chunk in doc_chunks)
    evidence: list[str] = []
    passed = True
    for needle in POLICY_CLAUSE_CHECKS:
        ok = needle in combined
        evidence.append(f"{needle}: {'PASS' if ok else 'FAIL'}")
        passed = passed and ok
    return FidelityResult(name="policy_fidelity", passed=passed, evidence=evidence)


def check_prospectus_fidelity(chunks: list[ChunkRecord], *, max_chunk_tokens: int) -> FidelityResult:
    evidence: list[str] = []
    passed = True
    for document_id in PROSPECTUS_DOC_IDS:
        doc_chunks = _chunks_for_document(chunks, document_id)
        if not doc_chunks:
            passed = False
            evidence.append(f"{document_id}: missing chunks")
            continue
        max_tokens = max(chunk.token_count for chunk in doc_chunks)
        giant = max_tokens > max_chunk_tokens
        metadata_leak = any("sha256:" in chunk.content.lower() for chunk in doc_chunks)
        hard_fallback = sum(1 for chunk in doc_chunks if chunk.split_method.value == "hard_token_fallback")
        missing_paths = sum(1 for chunk in doc_chunks if not chunk.section_path)
        passed = passed and not giant and not metadata_leak and len(doc_chunks) > 1
        evidence.append(
            f"{document_id}: chunks={len(doc_chunks)}, max_tokens={max_tokens}, "
            f"giant={giant}, metadata_leak={metadata_leak}, hard_fallback={hard_fallback}, "
            f"missing_section_paths={missing_paths}"
        )
    return FidelityResult(name="prospectus_fidelity", passed=passed, evidence=evidence)


def check_faq_atomicity(chunks: list[ChunkRecord]) -> FidelityResult:
    violations = []
    for chunk in chunks:
        if effective_document_type(chunk.document_type, chunk.title, chunk.canonical_url) != "faq":
            continue
        has_q = "Question:" in chunk.content
        has_a = "Answer:" in chunk.content
        if has_q and not has_a:
            violations.append(chunk.chunk_id)
    return FidelityResult(
        name="faq_atomicity",
        passed=not violations,
        evidence=[f"violations={len(violations)}"] + violations[:5],
    )
