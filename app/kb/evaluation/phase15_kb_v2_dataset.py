"""Phase 15 KB V2 retrieval evaluation dataset (9 questions)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.kb.embedding.phase13_corpus import Phase13Corpus, load_phase13_validated_corpus
from app.kb.ingestion.golden_audit import find_golden_match
from app.kb.ingestion.models import Phase12ChunkRecord

PHASE15_QUESTIONS: list[tuple[str, str]] = [
    ("ERK-V2-001", "what is this bte?"),
    ("ERK-V2-002", "who is return policy?"),
    ("ERK-V2-003", "what is the address of earkart?"),
    ("ERK-V2-004", "how to contact the earkart?"),
    ("ERK-V2-005", "when to except the deeeliver of item"),
    ("ERK-V2-006", "where is the office of earkart?"),
    ("ERK-V2-007", "what is the omni?"),
    ("ERK-V2-008", "what is the battery life ?"),
    ("ERK-V2-009", "why to buy from u ?"),
]

DATASET_NAME = "earkart_kb_v2_9"


@dataclass(frozen=True)
class ExpectedKnowledge:
    answerability: str
    expected_chunk_ids: list[str]
    expected_document_ids: list[str]
    expected_section_path: list[str]
    mapping_notes: str = ""
    status: str = "mapped"


def _chunks_for_doc(corpus: Phase13Corpus, label: str) -> list[Phase12ChunkRecord]:
    for doc in corpus.documents:
        if doc.label == label:
            return doc.chunks
    return []


def _all_chunks(corpus: Phase13Corpus) -> list[Phase12ChunkRecord]:
    return corpus.all_chunks


def _find_chunks(
    chunks: list[Phase12ChunkRecord],
    patterns: list[str],
    *,
    required_section: str | None = None,
    exclude_metadata_only: bool = False,
) -> list[Phase12ChunkRecord]:
    matched: list[Phase12ChunkRecord] = []
    for chunk in chunks:
        haystack = f"{chunk.content} {' '.join(chunk.section_path)}".lower()
        if required_section and not any(
            part.strip().startswith(required_section) for part in chunk.section_path
        ):
            continue
        if exclude_metadata_only and _is_source_pages_only(chunk.content):
            continue
        if any(pattern.lower() in haystack for pattern in patterns):
            matched.append(chunk)
    return matched


def _is_source_pages_only(content: str) -> bool:
    lowered = content.lower()
    return "source pages:" in lowered and lowered.count("http") >= 3 and len(content) > 400


def _pick_best(candidates: list[Phase12ChunkRecord]) -> Phase12ChunkRecord | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.token_count, item.chunk_id))[0]


def resolve_expected_knowledge(corpus: Phase13Corpus) -> dict[str, ExpectedKnowledge]:
    merged = _chunks_for_doc(corpus, "merged")
    terms = _chunks_for_doc(corpus, "terms")
    all_chunks = _all_chunks(corpus)
    resolved: dict[str, ExpectedKnowledge] = {}

    bte = find_golden_match(
        merged,
        document_name="merged",
        concept_id="bte",
        patterns=["behind-the-ear (bte)", "behind-the-ear", "bte"],
    )
    if bte:
        merged_doc = next(doc for doc in corpus.documents if doc.label == "merged")
        resolved["ERK-V2-001"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[bte["chunk_id"]],
            expected_document_ids=[merged_doc.record.document_id],
            expected_section_path=bte["section_path"],
            mapping_notes="BTE definition is in the hearing-aid types chunk under section 6.3 Bluup+.",
        )

    returns = find_golden_match(
        terms,
        document_name="terms",
        concept_id="returns_refunds",
        patterns=["return and refund", "returns, replacements", "return & refund"],
        required_section="7",
    )
    if returns:
        terms_doc = next(doc for doc in corpus.documents if doc.label == "terms")
        resolved["ERK-V2-002"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[returns["chunk_id"]],
            expected_document_ids=[terms_doc.record.document_id],
            expected_section_path=returns["section_path"],
            mapping_notes=(
                "Grammatically odd query preserved verbatim. Maps to section 7.1 Return & Refund policy reference."
            ),
        )

    address = find_golden_match(
        terms,
        document_name="terms",
        concept_id="legal_compliance",
        patterns=["registered office", "corporate office"],
    )
    if address:
        terms_doc = next(doc for doc in corpus.documents if doc.label == "terms")
        resolved["ERK-V2-003"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[address["chunk_id"]],
            expected_document_ids=[terms_doc.record.document_id],
            expected_section_path=address["section_path"],
            mapping_notes="Company Details block contains Registered Office and Corporate Office addresses.",
        )
        resolved["ERK-V2-006"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[address["chunk_id"]],
            expected_document_ids=[terms_doc.record.document_id],
            expected_section_path=address["section_path"],
            mapping_notes="Office locations are in the same Company Details chunk as registered/corporate office addresses.",
        )

    contact_candidates = _find_chunks(
        all_chunks,
        [
            "contact us",
            "customer support",
            "grievance",
            "reach us",
            "phone",
            "email",
            "call us",
            "support@",
            "@earkart",
        ],
    )
    contact = _pick_best(contact_candidates)
    if contact:
        resolved["ERK-V2-004"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[contact.chunk_id],
            expected_document_ids=[contact.document_id],
            expected_section_path=contact.section_path,
            mapping_notes="Best corpus match for contact/support information (no dedicated phone/email block found).",
        )
    else:
        resolved["ERK-V2-004"] = ExpectedKnowledge(
            answerability="CORPUS_GAP",
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_section_path=[],
            mapping_notes="No explicit contact phone/email or grievance section found in Phase 13 PDF corpus.",
            status="corpus_gap",
        )

    delivery_candidates = _find_chunks(
        terms + merged,
        [
            "delivery",
            "deliver",
            "dispatch",
            "shipping",
            "shipment",
            "expected delivery",
            "time of delivery",
        ],
    )
    delivery = _pick_best(delivery_candidates)
    if delivery:
        resolved["ERK-V2-005"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[delivery.chunk_id],
            expected_document_ids=[delivery.document_id],
            expected_section_path=delivery.section_path,
            mapping_notes="Query spelling preserved ('except/deeeliver'). Mapped to best delivery-related corpus chunk.",
        )
    else:
        resolved["ERK-V2-005"] = ExpectedKnowledge(
            answerability="CORPUS_GAP",
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_section_path=[],
            mapping_notes="No delivery timing or shipping expectation section found in Phase 13 PDF corpus.",
            status="corpus_gap",
        )

    omni_candidates = [
        chunk
        for chunk in merged
        if re.search(r"\bomni\b", chunk.content, re.I)
        and not _is_source_pages_only(chunk.content)
    ]
    omni = _pick_best(omni_candidates)
    if omni:
        merged_doc = next(doc for doc in corpus.documents if doc.label == "merged")
        resolved["ERK-V2-007"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[omni.chunk_id],
            expected_document_ids=[merged_doc.record.document_id],
            expected_section_path=omni.section_path,
            mapping_notes="OMNI content match excluding source-page metadata preamble only.",
        )
    else:
        resolved["ERK-V2-007"] = ExpectedKnowledge(
            answerability="CORPUS_GAP",
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_section_path=[],
            mapping_notes="OMNI appears only in merged extract metadata/source pages, not as standalone explanatory content.",
            status="corpus_gap",
        )

    battery_life_candidates = _find_chunks(
        all_chunks,
        ["battery life", "hours of use", "battery lasts", "battery duration"],
    )
    if battery_life_candidates:
        battery = _pick_best(battery_life_candidates)
        resolved["ERK-V2-008"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[battery.chunk_id],
            expected_document_ids=[battery.document_id],
            expected_section_path=battery.section_path,
            mapping_notes="Direct battery-life specification found in corpus.",
        )
    else:
        resolved["ERK-V2-008"] = ExpectedKnowledge(
            answerability="CORPUS_GAP",
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_section_path=[],
            mapping_notes=(
                "Corpus mentions rechargeable batteries and free batteries but no battery-life duration/spec."
            ),
            status="corpus_gap",
        )

    benefits = find_golden_match(
        merged,
        document_name="merged",
        concept_id="benefits",
        patterns=["why choose earkart", "why choose earKART", "what makes us better"],
    )
    if benefits:
        merged_doc = next(doc for doc in corpus.documents if doc.label == "merged")
        resolved["ERK-V2-009"] = ExpectedKnowledge(
            answerability="ANSWERABLE",
            expected_chunk_ids=[benefits["chunk_id"]],
            expected_document_ids=[merged_doc.record.document_id],
            expected_section_path=benefits["section_path"],
            mapping_notes="Maps to 'Why Choose earKART' / benefits section.",
        )

    for query_id, _ in PHASE15_QUESTIONS:
        resolved.setdefault(
            query_id,
            ExpectedKnowledge(
                answerability="CORPUS_GAP",
                expected_chunk_ids=[],
                expected_document_ids=[],
                expected_section_path=[],
                mapping_notes="Could not resolve expected knowledge from Phase 13 corpus.",
                status="corpus_gap",
            ),
        )
    return resolved


def build_phase15_cases(corpus: Phase13Corpus | None = None) -> list[dict[str, Any]]:
    corpus = corpus or load_phase13_validated_corpus()
    expected_by_id = resolve_expected_knowledge(corpus)
    cases: list[dict[str, Any]] = []
    for query_id, question in PHASE15_QUESTIONS:
        expected = expected_by_id[query_id]
        cases.append(
            {
                "id": query_id,
                "query": question,
                "question": question,
                "answerability": expected.answerability,
                "expected_chunk_ids": expected.expected_chunk_ids,
                "expected_document_ids": expected.expected_document_ids,
                "expected_section_path": expected.expected_section_path,
                "mapping_notes": expected.mapping_notes,
                "expected_status": expected.status,
            }
        )
    return cases
