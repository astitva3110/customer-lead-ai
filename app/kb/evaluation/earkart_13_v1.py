"""Frozen Earkart 13-question retrieval evaluation set (manual ground truth)."""

from __future__ import annotations

from typing import Any

EVALUATION_DATASET_VERSION = "earkart_13_v1"

# Ground truth curated from frozen production chunks — not keyword overlap alone.
EARKART_13_CASES: list[dict[str, Any]] = [
    {
        "id": "ERK-001",
        "query": "What is Bluup?",
        "expected_chunk_ids": [
            "0a2c53d2d6764d8267d9f1194ad3720a414afe783aebdf1510ef9054fa3f4a8e",
            "9feb31aa73c4c4e98121710f5452ed42e4412e4fbbdad16682ad2f835a50e060",
        ],
        "expected_document_id": "782e0f59-837c-50e9-b2c3-e49da2974807",
        "expected_knowledge": "Bluup+ is an earKART reusable noise-protection earplug product with attenuation filters.",
        "category": "product",
    },
    {
        "id": "ERK-002",
        "query": "What is the price of Bluup?",
        "expected_chunk_ids": [
            "e126d16f83b11a9f2a2edf98e5562164440ea4bfb5527b8115d419704f6c3c72",
            "9082b44bd460591be300b660121dffe39d6cd899d7d9331e49c7c49c24d80226",
        ],
        "expected_document_id": "782e0f59-837c-50e9-b2c3-e49da2974807",
        "expected_knowledge": "Bluup+ sale price Rs. 1,490; standard Bluup Rs. 999 on product pages.",
        "category": "product_pricing",
    },
    {
        "id": "ERK-003",
        "query": "What is Earkart?",
        "expected_chunk_ids": [
            "8017d4bedf0d96cbd4c413b0536430e2581a58dafb23e6fdefe4cd2c866ae542",
        ],
        "expected_document_id": "aeeb4794-9477-5a1c-913f-aa274af6aa7c",
        "expected_knowledge": "earKART is India's leading digital-first hearing care platform founded by Rohit Misra.",
        "category": "company",
    },
    {
        "id": "ERK-004",
        "query": "Who is Rohit Mirsa?",
        "expected_chunk_ids": [
            "e762e14aa9b176e9f6dadb7bac20e78481acfc8ae3ebcc5419b1a37d3b3a1699",
            "8017d4bedf0d96cbd4c413b0536430e2581a58dafb23e6fdefe4cd2c866ae542",
            "4755cc2acd4449150a9b5741441a600a162a79d32def076712a0e908815cfb32",
        ],
        "expected_document_id": "8d0f629d-4310-566b-8ff5-5921342f2696",
        "expected_knowledge": "Rohit Misra is Promoter/Managing Director of Earkart with long hearing-healthcare experience.",
        "category": "people",
        "query_note": "User query misspells Misra as Mirsa.",
    },
    {
        "id": "ERK-005",
        "query": "What are the different types of hearing aids?",
        "expected_chunk_ids": [
            "ccbe719ce8237278985d7d9fb4f5f4c8d824dd959b507c1ae555a19bce2c7f43",
            "7ae900f52dec4db810ee9e0ab3b2b2d1a0619aff4b658eebd9afd849bf501e8b",
            "33f51e30a87623aee0fde8cdb2ac9c86d127d03dfdc0a1fabe60d1b535676d92",
            "441c45d43e710ac9f9f85fedc1ba04780a5313f5af756e753fa084fffa348d2f",
        ],
        "expected_document_id": "dbd99379-405b-5462-aa88-ef623991666b",
        "expected_knowledge": "Hearing aid types include BTE, RIC, ITE, CIC and related categories on hearing-aids page.",
        "category": "product_education",
    },
    {
        "id": "ERK-006",
        "query": "What is Signia?",
        "expected_chunk_ids": [],
        "expected_document_id": None,
        "expected_knowledge": "Signia brand information is not present in the frozen KB corpus.",
        "category": "product_brand",
        "failure_category_default": "SOURCE_DATA",
    },
    {
        "id": "ERK-007",
        "query": "What is TINY?",
        "expected_chunk_ids": [
            "6d50dd10d207f383854f8635427be4b17024a9a38c8bd85e5b6a0be19f201756",
        ],
        "expected_document_id": "3547bffa-3bfb-5ffd-8288-473809f06ebd",
        "expected_knowledge": "TINY is a rechargeable CIC hearing aid product sold by earKART.",
        "category": "product",
    },
    {
        "id": "ERK-008",
        "query": "Name the board directors.",
        "expected_chunk_ids": [
            "e762e14aa9b176e9f6dadb7bac20e78481acfc8ae3ebcc5419b1a37d3b3a1699",
            "fe9310ef35f5fbfef2b6e78f7a8fcca1cec6ef8ca4a72d44208e61e52c09b14b",
            "c8ea096c9e8b51746a21f1254bfa033f96fd8198a4dccde6454bf0e206612836",
            "a97dd15e29613309c8372f1314780eb302309478b55898b232c2bf180d9dcc61",
        ],
        "expected_document_id": "8d0f629d-4310-566b-8ff5-5921342f2696",
        "expected_knowledge": "Board includes Rohit Misra, Monika Misra, Ajay Kumar Giri, Rahul Salesha and independent directors.",
        "category": "governance",
    },
    {
        "id": "ERK-009",
        "query": "How does a hearing aid help with hearing loss?",
        "expected_chunk_ids": [
            "efba46e8c63f9acf7e111e7690e1ef52982091eea19ec604ac6bf1250d1cd06f",
            "b7e8824be76badfee3caf473173beaf508584320864c974765ff8304a277ef07",
        ],
        "expected_document_id": "8bd45b19-cb3d-5e56-ad5a-b67f1af7d2f9",
        "expected_knowledge": "Hearing aids amplify sound via microphone, amplifier, and speaker to help people hear better.",
        "category": "product_education",
    },
    {
        "id": "ERK-010",
        "query": "What are Analog and Digital hearing aids?",
        "expected_chunk_ids": [
            "abf570d779649497f2689a0dbd39a12482999dc7dbed5a6261013c219d73b8a2",
            "dc81fba0071ef399e9e37b9e41b9b389c0f3357e96c619010a5eed7652e07773",
            "94a7d04cb15c915c1f08058751878dc8d0164a0aa57d9f46cb04e27d40e4ab9a",
        ],
        "expected_document_id": "90e7162c-fe93-5e88-b2ca-e30a890e0f81",
        "expected_knowledge": "Analog vs digital hearing aid electronics explained in FAQ and prospectus content.",
        "category": "product_education",
    },
    {
        "id": "ERK-011",
        "query": "What distinguishes an amplifier from a hearing aid?",
        "expected_chunk_ids": [
            "9fa70d411f56e710098a04c9115e21a5044c3432da92a5d71197682cbb7e7efa",
        ],
        "expected_document_id": "187d654e-7ceb-5ae1-ba2f-65561956c41d",
        "expected_knowledge": "PSAPs/amplifiers boost all sounds; hearing aids adjust to individual hearing profiles.",
        "category": "product_education",
    },
    {
        "id": "ERK-012",
        "query": "How do hearing aids work?",
        "expected_chunk_ids": [
            "0e1f4c23e321103435c597b1ac97694956a86eb545043b82d7a2c9c1d07e79b5",
            "4274167974b250dfe2eca746ed9c45a141907066ecd8d18f2f83cdfc76f0a44a",
        ],
        "expected_document_id": "01e76cb2-317f-509a-b8b5-f473f5bd5400",
        "expected_knowledge": "Hearing aids use microphone, amplifier, and speaker to capture, boost, and deliver sound.",
        "category": "product_education",
    },
    {
        "id": "ERK-013",
        "query": "What are the timings of the customer support center?",
        "expected_chunk_ids": [
            "329454b63bdfbca93d00fcf7cf518824c2a87804d0307019cc0deba58279607a",
        ],
        "expected_document_id": "90e7162c-fe93-5e88-b2ca-e30a890e0f81",
        "expected_knowledge": "Customer support is active 10:00am to 6:00pm; Sunday closed.",
        "category": "support",
    },
]
