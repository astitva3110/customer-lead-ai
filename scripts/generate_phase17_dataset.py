#!/usr/bin/env python3
"""Generate Phase 17 comprehensive evaluation dataset."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluations" / "earkart_kb_v3_1_comprehensive.json"

QUESTIONS = [
    # PRODUCT
    ("ERK-C01", "What is BTE?", "PRODUCT", "product_definition", "ANSWERABLE", {
        "knowledge_key": "hearing_aid_type:bte", "document_label": "merged",
        "section_path_contains": ["Behind-The-Ear"],
        "content_patterns": ["behind-the-ear (bte)", "behind-the-ear", "bte"],
    }),
    ("ERK-C02", "What is TINY?", "PRODUCT", "product_definition", "ANSWERABLE", {
        "knowledge_key": "product:tiny", "document_label": "merged",
        "section_path_contains": ["6.1 TINY"],
        "content_patterns": ["tiny"],
    }),
    ("ERK-C03", "What is OMNI?", "PRODUCT", "product_definition", "CORPUS_GAP", {
        "knowledge_key": "product:omni",
    }),
    ("ERK-C04", "What is Bluup?", "PRODUCT", "product_definition", "ANSWERABLE", {
        "knowledge_key": "product:bluup", "document_label": "merged",
        "section_path_contains": ["6.2 Bluup"],
        "content_patterns": ["bluup"],
    }),
    ("ERK-C05", "Tell me about Bluup.", "PRODUCT", "product_overview", "ANSWERABLE", {
        "knowledge_key": "product:bluup_overview", "document_label": "merged",
        "section_path_contains": ["6.2 Bluup"],
        "content_patterns": ["bluup", "earplugs", "noise-reducing"],
    }),
    ("ERK-C06", "What are the different types of hearing aids?", "PRODUCT", "product_taxonomy", "ANSWERABLE", {
        "knowledge_key": "hearing_aid:types", "document_label": "merged",
        "section_path_contains": ["Types of Hearing Aids"],
        "content_type": "hearing_aid_type",
    }),
    # SPECIFICATION
    ("ERK-C07", "What is the battery life of the product?", "SPECIFICATION", "product_spec", "CORPUS_GAP", {
        "knowledge_key": "product:battery_life_duration",
    }),
    ("ERK-C08", "What features does TINY have?", "SPECIFICATION", "product_features", "ANSWERABLE", {
        "knowledge_key": "product:tiny_features", "document_label": "merged",
        "section_path_contains": ["6.1 TINY"],
        "content_patterns": ["features:", "rechargeable", "16-channel"],
        "content_type": "product_specification",
    }),
    ("ERK-C09", "What features does OMNI have?", "SPECIFICATION", "product_features", "CORPUS_GAP", {
        "knowledge_key": "product:omni_features",
    }),
    ("ERK-C10", "What is special about Bluup?", "SPECIFICATION", "product_features", "ANSWERABLE", {
        "knowledge_key": "product:bluup_plus_features", "document_label": "merged",
        "section_path_contains": ["6.3 Bluup+"],
        "content_patterns": ["digital hearing", "amplifier", "microphone"],
    }),
    ("ERK-C11", "How does BTE work?", "SPECIFICATION", "product_mechanism", "ANSWERABLE", {
        "knowledge_key": "faq:how_hearing_aid_works", "document_label": "merged",
        "content_patterns": ["how does a hearing aid work", "three main parts", "microphone"],
        "content_type": "faq_pair",
    }),
    ("ERK-C12", "What is the difference between BTE and other hearing aids?", "SPECIFICATION", "product_comparison", "ANSWERABLE", {
        "knowledge_key": "hearing_aid:type_comparison", "document_label": "merged",
        "section_path_contains": ["Types of Hearing Aids"],
        "content_type": "hearing_aid_type",
    }),
    # HEARING_AID
    ("ERK-C13", "How do hearing aids work?", "HEARING_AID", "general_knowledge", "ANSWERABLE", {
        "knowledge_key": "faq:how_hearing_aid_works", "document_label": "merged",
        "content_patterns": ["how does a hearing aid work", "three main parts"],
        "content_type": "faq_pair",
    }),
    ("ERK-C14", "How does a hearing aid help with hearing loss?", "HEARING_AID", "general_knowledge", "ANSWERABLE", {
        "knowledge_key": "faq:hearing_loss_help", "document_label": "merged",
        "content_patterns": ["help with hearing loss", "sensorineural hearing loss"],
        "content_type": "faq_pair",
    }),
    ("ERK-C15", "What are analog and digital hearing aids?", "HEARING_AID", "general_knowledge", "ANSWERABLE", {
        "knowledge_key": "faq:analog_vs_digital", "document_label": "merged",
        "content_patterns": ["analog vs. digital", "analog and digital"],
        "content_type": "faq_pair",
    }),
    ("ERK-C16", "What distinguishes an amplifier from a hearing aid?", "HEARING_AID", "general_knowledge", "ANSWERABLE", {
        "knowledge_key": "faq:amplifier_vs_aid", "document_label": "merged",
        "content_patterns": ["hearing amplifier from a hearing aid", "distinguishes a hearing amplifier"],
        "content_type": "faq_pair",
    }),
    ("ERK-C17", "What are the different hearing aid styles?", "HEARING_AID", "general_knowledge", "ANSWERABLE", {
        "knowledge_key": "hearing_aid:styles", "document_label": "merged",
        "section_path_contains": ["Types of Hearing Aids"],
        "content_type": "hearing_aid_type",
    }),
    # COMPANY
    ("ERK-C18", "What is Earkart?", "COMPANY", "company_overview", "ANSWERABLE", {
        "knowledge_key": "company:overview", "document_label": "merged",
        "content_patterns": ["earkart", "india's largest network", "1500+ hearing aid clinics"],
        "content_type": "contact_information",
    }),
    ("ERK-C19", "What is the address of Earkart?", "COMPANY", "company_address", "ANSWERABLE", {
        "knowledge_key": "company:registered_office", "document_label": "terms",
        "content_patterns": ["registered office", "corporate office"],
    }),
    ("ERK-C20", "Where is the Earkart office?", "COMPANY", "company_address", "ANSWERABLE", {
        "knowledge_key": "company:office_location", "document_label": "terms",
        "content_patterns": ["registered office", "corporate office"],
    }),
    ("ERK-C21", "How can I contact Earkart?", "COMPANY", "contact_info", "ANSWERABLE", {
        "knowledge_key": "support:contact",
        "content_patterns": ["info@earkart.com", "phone:", "contact", "9289097578"],
    }),
    ("ERK-C22", "What are the customer support timings?", "COMPANY", "support_hours", "ANSWERABLE", {
        "knowledge_key": "support:timings", "document_label": "merged",
        "content_patterns": ["customer support center timings", "10:00 am", "monday–saturday"],
    }),
    # POLICY
    ("ERK-C23", "What is the return policy?", "POLICY", "returns", "ANSWERABLE", {
        "knowledge_key": "policy:returns_refunds", "document_label": "terms",
        "required_section_prefix": "7",
        "content_patterns": ["return and refund", "returns, replacements"],
    }),
    ("ERK-C24", "When can I return an item?", "POLICY", "returns", "ANSWERABLE", {
        "knowledge_key": "policy:return_eligibility", "document_label": "terms",
        "section_path_contains": ["7.3 Return Condition Requirements"],
        "content_patterns": ["return", "eligible for return"],
    }),
    ("ERK-C25", "When should I expect delivery?", "POLICY", "delivery", "ANSWERABLE", {
        "knowledge_key": "policy:delivery_timeline", "document_label": "terms",
        "section_path_contains": ["4.5 Delivery Timeline Variance"],
        "content_patterns": ["delivery timeline"],
        "exclude_patterns": ["delivers the amplified sound"],
    }),
    ("ERK-C26", "How long does delivery take?", "POLICY", "delivery", "ANSWERABLE", {
        "knowledge_key": "policy:delivery_duration", "document_label": "terms",
        "section_path_contains": ["4.5 Delivery Timeline Variance"],
        "content_patterns": ["delivery timeline"],
        "exclude_patterns": ["delivers the amplified sound"],
    }),
    ("ERK-C27", "What is the warranty policy?", "POLICY", "warranty", "ANSWERABLE", {
        "knowledge_key": "policy:warranty", "document_label": "terms",
        "required_section_prefix": "6",
        "content_patterns": ["warranty"],
    }),
    ("ERK-C28", "What happens if my order is cancelled?", "POLICY", "cancellation", "ANSWERABLE", {
        "knowledge_key": "policy:order_cancellation", "document_label": "terms",
        "section_path_contains": ["5.3 Order Cancellation"],
        "content_patterns": ["cancelled", "cancellation"],
    }),
    ("ERK-C29", "What are the non-returnable categories?", "POLICY", "returns", "ANSWERABLE", {
        "knowledge_key": "policy:non_returnable", "document_label": "terms",
        "section_path_contains": ["7.4 Non-Returnable Categories"],
        "content_patterns": ["non-returnable"],
    }),
    ("ERK-C30", "What are the payment terms?", "POLICY", "payment", "ANSWERABLE", {
        "knowledge_key": "policy:payment", "document_label": "terms",
        "section_path_contains": ["5.4 Payment Method"],
        "content_patterns": ["payment", "payment gateways"],
    }),
    # SUMMARY
    ("ERK-C31", "Why should I buy from Earkart?", "SUMMARY", "benefits_summary", "ANSWERABLE", {
        "knowledge_key": "benefits:why_choose_summary", "document_label": "merged",
        "section_path_contains": ["Why Choose earKART"],
        "content_patterns": ["why choose earkart", "free insurance"],
    }),
    ("ERK-C32", "Why should I choose Earkart?", "SUMMARY", "benefits_summary", "ANSWERABLE", {
        "knowledge_key": "benefits:why_choose_summary", "document_label": "merged",
        "section_path_contains": ["Why Choose earKART"],
        "content_patterns": ["why choose earkart", "free insurance"],
    }),
    ("ERK-C33", "What are the benefits of buying from Earkart?", "SUMMARY", "benefits_summary", "ANSWERABLE", {
        "knowledge_key": "benefits:why_choose_summary", "document_label": "merged",
        "section_path_contains": ["Why Choose earKART"],
        "content_patterns": ["benefits", "free insurance", "why choose"],
    }),
    ("ERK-C34", "Tell me about Earkart's products.", "SUMMARY", "catalog_overview", "ANSWERABLE", {
        "knowledge_key": "company:products_overview", "document_label": "merged",
        "content_patterns": ["what makes us better", "products from global manufacturers", "tiny", "bluup"],
    }),
    ("ERK-C35", "Tell me about Earkart's hearing aids.", "SUMMARY", "catalog_overview", "ANSWERABLE", {
        "knowledge_key": "hearing_aid:catalog_overview", "document_label": "merged",
        "section_path_contains": ["Types of Hearing Aids"],
        "content_type": "hearing_aid_type",
    }),
    # NOISY
    ("ERK-C36", "what is this bte?", "NOISY_QUERY", "noisy_product", "ANSWERABLE", {
        "knowledge_key": "hearing_aid_type:bte", "document_label": "merged",
        "section_path_contains": ["Behind-The-Ear"],
        "content_patterns": ["behind-the-ear (bte)", "bte"],
    }),
    ("ERK-C37", "what is the omni?", "NOISY_QUERY", "noisy_product", "CORPUS_GAP", {
        "knowledge_key": "product:omni",
    }),
    ("ERK-C38", "what is earkart?", "NOISY_QUERY", "noisy_company", "ANSWERABLE", {
        "knowledge_key": "company:overview", "document_label": "merged",
        "content_patterns": ["earkart", "1500+ hearing aid clinics"],
        "content_type": "contact_information",
    }),
    ("ERK-C39", "where is the office of earkart?", "NOISY_QUERY", "noisy_company", "ANSWERABLE", {
        "knowledge_key": "company:office_location", "document_label": "terms",
        "content_patterns": ["registered office", "corporate office"],
    }),
    ("ERK-C40", "when to except the deeeliver of item", "NOISY_QUERY", "noisy_policy", "ANSWERABLE", {
        "knowledge_key": "policy:delivery_timeline", "document_label": "terms",
        "section_path_contains": ["4.5 Delivery Timeline Variance"],
        "content_patterns": ["delivery timeline"],
        "exclude_patterns": ["delivers the amplified sound"],
    }),
    ("ERK-C41", "why to buy from u?", "NOISY_QUERY", "noisy_summary", "ANSWERABLE", {
        "knowledge_key": "benefits:why_choose_summary", "document_label": "merged",
        "section_path_contains": ["Why Choose earKART"],
        "content_patterns": ["why choose earkart", "free insurance"],
    }),
    ("ERK-C42", "how hearing aid helps in hearing loos?", "NOISY_QUERY", "noisy_hearing_aid", "ANSWERABLE", {
        "knowledge_key": "faq:hearing_loss_help", "document_label": "merged",
        "content_patterns": ["help with hearing loss", "sensorineural"],
        "content_type": "faq_pair",
    }),
    # CORPUS GAP
    ("ERK-C43", "What is the battery life in hours?", "CORPUS_GAP", "missing_spec", "CORPUS_GAP", {
        "knowledge_key": "product:battery_life_hours",
    }),
    ("ERK-C44", "What is the Signia hearing aid?", "CORPUS_GAP", "missing_product", "CORPUS_GAP", {
        "knowledge_key": "product:signia",
    }),
]


def main() -> None:
    payload = {
        "dataset": "earkart_kb_v3_1_comprehensive",
        "questions": [
            {
                "question_id": qid,
                "question": question,
                "category": category,
                "expected_answer_type": answer_type,
                "answerability": answerability,
                "expected": expected,
            }
            for qid, question, category, answer_type, answerability, expected in QUESTIONS
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(QUESTIONS)} questions)")


if __name__ == "__main__":
    main()
