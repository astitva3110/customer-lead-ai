"""35-question production chunk coverage evaluation."""

from __future__ import annotations

from typing import Any

from app.kb.chunking.models import ChunkRecord
from app.kb.chunking.semantic_units import effective_document_type

RETURNS_DOC_ID = "5f692faa-9f76-5643-9c78-b749258d06b4"
RADIUS_DOC_ID = "5f2ec8ed-c57d-5511-8d23-df8adf1b548d"
PROSPECTUS_IDS = (
    "01e76cb2-317f-509a-b8b5-f473f5bd5400",
    "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9",
)

QUESTION_COVERAGE: list[dict[str, Any]] = [
    {"id": "QC01", "topic": "return_policy", "question": "What is the return period for hearing aids?", "needles": ["ten (10) calendar days", "1.1.1"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC02", "topic": "return_policy", "question": "What is the return period for earplugs?", "needles": ["seven (7) calendar days", "1.2.1"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC03", "topic": "return_policy", "question": "What are earplug return eligibility requirements?", "needles": ["1.2.2 Eligibility", "hygiene"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC04", "topic": "return_policy", "question": "Must returns include invoice or receipt?", "needles": ["2.1", "valid invoice/receipt"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC05", "topic": "return_policy", "question": "Who pays return shipping?", "needles": ["2.6", "Return shipping costs"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC06", "topic": "return_policy", "question": "How do I start a return?", "needles": ["3.1", "info@earkart.com"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC07", "topic": "return_policy", "question": "What is RMA?", "needles": ["3.2", "Return Merchandise Authorization"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC08", "topic": "return_policy", "question": "Where do I ship returns?", "needles": ["3.3", "Sector 62, Noida"], "doc_id": RETURNS_DOC_ID, "require_single_chunk": True},
    {"id": "QC09", "topic": "return_policy", "question": "How long for refund after approval?", "needles": ["4.1", "seven (7) to ten (10) business days"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC10", "topic": "return_policy", "question": "Are statutory consumer rights preserved?", "needles": ["5. Statutory Rights", "statutory rights"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC11", "topic": "replacement_policy", "question": "How long for replacement delivery?", "needles": ["2.7", "15 days"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC12", "topic": "replacement_policy", "question": "Can promotional BOGO items be refunded?", "needles": ["4.3", "non-refundable"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC13", "topic": "product_spec", "question": "Radius M16 BTE battery life?", "needles": ["270 Hrs", "Battery Life"], "doc_id": RADIUS_DOC_ID},
    {"id": "QC14", "topic": "product_spec", "question": "Radius M16 BTE battery size?", "needles": ["Battery Size", "13"], "doc_id": RADIUS_DOC_ID},
    {"id": "QC15", "topic": "product_spec", "question": "Radius M16 attack time?", "needles": ["Attack Time 28 ms"], "doc_id": RADIUS_DOC_ID},
    {"id": "QC16", "topic": "product_spec", "question": "Radius M16 release time?", "needles": ["Release Time 891 ms"], "doc_id": RADIUS_DOC_ID},
    {"id": "QC17", "topic": "product_spec", "question": "Radius M16 fitting range?", "needles": ["Fitting Range", "30 -120 dB"], "doc_id": RADIUS_DOC_ID},
    {"id": "QC18", "topic": "shipping", "question": "Return shipping deadline after approval?", "needles": ["2.3", "five (5) business days"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC19", "topic": "warranty", "question": "Are unauthorized reseller products eligible?", "needles": ["2.4", "unauthorized resellers"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC20", "topic": "warranty", "question": "What voids return eligibility for damage?", "needles": ["2.2", "misuse, physical damage"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC21", "topic": "faq", "question": "FAQ battery life answer?", "needles": ["Question:", "Answer:", "battery"], "doc_type": "faq"},
    {"id": "QC22", "topic": "faq", "question": "FAQ content atomicity?", "needles": ["Question:", "Answer:"], "doc_type": "faq"},
    {"id": "QC23", "topic": "investor", "question": "Investor IPO prospectus risk disclosure exists?", "needles": ["risk factors", "Investments in equity"], "doc_id": PROSPECTUS_IDS[0]},
    {"id": "QC24", "topic": "investor", "question": "Final prospectus company address?", "needles": ["Sector 63", "Noida"], "doc_id": PROSPECTUS_IDS[1]},
    {"id": "QC25", "topic": "investor", "question": "Draft prospectus offer document present?", "needles": ["DRHP", "Public Issue"], "doc_id": PROSPECTUS_IDS[0]},
    {"id": "QC26", "topic": "pdf", "question": "PDF page provenance on Radius spec sheet?", "needles": ["Battery Life"], "doc_id": RADIUS_DOC_ID, "require_page": True},
    {"id": "QC27", "topic": "ocr", "question": "OCR due diligence certificate content?", "needles": ["Due Diligence Certificate", "SEBI"], "extraction_method": "ocr"},
    {"id": "QC28", "topic": "ocr", "question": "OCR document retains meaningful certification text?", "needles": ["certify", "Prospectus"], "extraction_method": "ocr"},
    {"id": "QC29", "topic": "prospectus", "question": "Prospectus section hierarchy preserved?", "needles": ["SECTION", "GENERAL"], "doc_id": PROSPECTUS_IDS[0]},
    {"id": "QC30", "topic": "prospectus", "question": "Prospectus manufacturing facility risk factor?", "needles": ["Manufacturing Facility", "Noida"], "doc_id": PROSPECTUS_IDS[0]},
    {"id": "QC31", "topic": "company", "question": "Earkart contact email in policy?", "needles": ["info@earkart.com"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC32", "topic": "company", "question": "Earkart phone support for returns?", "needles": ["9289097578"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC33", "topic": "replacement_policy", "question": "Inspection timeline after product receipt?", "needles": ["3.4", "five (5) business days"], "doc_id": RETURNS_DOC_ID},
    {"id": "QC34", "topic": "product_spec", "question": "Radius M16 frequency range?", "needles": ["Frequency Range", "250 Hz"], "doc_id": RADIUS_DOC_ID},
    {"id": "QC35", "topic": "investor", "question": "Prospectus promoter information?", "needles": ["PROMOTERS", "ROHIT MISRA"], "doc_id": PROSPECTUS_IDS[1]},
]


def _scope_chunks(chunks: list[ChunkRecord], q: dict) -> list[ChunkRecord]:
    scope = chunks
    if q.get("doc_id"):
        scope = [c for c in chunks if c.document_id == q["doc_id"]]
    if q.get("doc_type"):
        scope = [
            c
            for c in chunks
            if effective_document_type(c.document_type, c.title, c.canonical_url) == q["doc_type"]
        ]
    if q.get("extraction_method"):
        scope = [c for c in scope if c.extraction_method.value == q["extraction_method"]]
    return scope


def _match_question(
    q: dict,
    production: list[ChunkRecord],
    *,
    review_excluded: list[ChunkRecord] | None = None,
) -> dict[str, Any]:
    scope = _scope_chunks(production, q)
    matched_chunks: list[str] = []
    for chunk in scope:
        if all(needle.lower() in chunk.content.lower() for needle in q["needles"]):
            matched_chunks.append(chunk.chunk_id)

    combined = "\n".join(c.content for c in scope)
    knowledge_in_production = all(needle.lower() in combined.lower() for needle in q["needles"])
    single_production_chunk = bool(matched_chunks)

    page_ok = True
    if q.get("require_page") and matched_chunks:
        chunk = next(c for c in scope if c.chunk_id == matched_chunks[0])
        page_ok = chunk.page_number is not None

    coverage_failure = False
    if not knowledge_in_production and review_excluded:
        fallback_scope = _scope_chunks(review_excluded, q)
        fallback_combined = "\n".join(c.content for c in fallback_scope)
        if all(needle.lower() in fallback_combined.lower() for needle in q["needles"]):
            coverage_failure = True

    require_single = q.get("require_single_chunk", False)
    passed = knowledge_in_production and page_ok and (single_production_chunk if require_single else True)

    return {
        "id": q["id"],
        "topic": q["topic"],
        "question": q["question"],
        "passed": passed,
        "knowledge_present_in_production": knowledge_in_production,
        "single_production_chunk": single_production_chunk,
        "matched_chunk_ids": matched_chunks[:3],
        "inappropriate_split": knowledge_in_production and not single_production_chunk and require_single,
        "page_provenance_ok": page_ok,
        "question_coverage_failure": coverage_failure,
    }


def evaluate_question_coverage(
    production: list[ChunkRecord],
    *,
    review_excluded: list[ChunkRecord] | None = None,
) -> dict[str, Any]:
    results = [
        _match_question(q, production, review_excluded=review_excluded)
        for q in QUESTION_COVERAGE
    ]
    passed_count = sum(1 for r in results if r["passed"])
    coverage_failures = [r for r in results if r.get("question_coverage_failure")]
    return {
        "total": len(results),
        "passed": passed_count,
        "all_passed": passed_count == len(results) and not coverage_failures,
        "coverage_failures": len(coverage_failures),
        "results": results,
    }
