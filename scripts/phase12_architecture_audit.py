#!/usr/bin/env python3
"""Phase 12 read-only architecture audit — classifies KB modules before migration."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# fmt: off
INVENTORY: dict[str, list[dict[str, str]]] = {
    "A_scraping": [
        {"path": "app/helpers/crawler.py", "purpose": "Crawl4AI filter chains and crawl orchestration", "classification": "REMOVE"},
        {"path": "app/core/services/crawl_service.py", "purpose": "Async site crawl + dual-write to legacy KB", "classification": "REMOVE"},
        {"path": "app/services/crawler.py", "purpose": "Re-export alias for CrawlService", "classification": "REMOVE"},
        {"path": "app/presentation/routes/crawl.py", "purpose": "HTTP POST /crawl endpoints", "classification": "REMOVE"},
        {"path": "app/presentation/schemas/crawl.py", "purpose": "Crawl request/response DTOs", "classification": "REMOVE"},
        {"path": "scripts/crawl_site.py", "purpose": "CLI crawl entrypoint", "classification": "REMOVE"},
        {"path": "scripts/crawl_earkart.py", "purpose": "Earkart multi-site crawl wrapper", "classification": "REMOVE"},
        {"path": "scripts/complete_earkart_in.py", "purpose": "OCR PDFs + scrape missing HTML", "classification": "REMOVE"},
        {"path": "scripts/extract_pdfs.py", "purpose": "Extract PDFs from crawled HTML", "classification": "REMOVE"},
        {"path": "scripts/phase4_unscraped_report.py", "purpose": "Unscraped document report", "classification": "REMOVE"},
        {"path": "scripts/count_unscraped_pdfs.py", "purpose": "Count unscraped PDFs", "classification": "REMOVE"},
        {"path": "app/infrastructure/repositories/knowledge_repository.py", "purpose": "Legacy file-backed crawl storage", "classification": "REMOVE"},
        {"path": "app/services/knowledge_store.py", "purpose": "Alias for FileKnowledgeRepository", "classification": "REMOVE"},
        {"path": "app/core/interfaces/repositories/knowledge_repository.py", "purpose": "Legacy knowledge repo protocol", "classification": "REMOVE"},
        {"path": "app/core/services/chat_service.py", "purpose": "Keyword chat over crawled pages", "classification": "REFACTOR"},
        {"path": "app/presentation/routes/knowledge.py", "purpose": "Legacy knowledge stats endpoint", "classification": "REMOVE"},
        {"path": "app/helpers/tokenize.py", "purpose": "Keyword tokenization for legacy chat", "classification": "REMOVE"},
        {"path": "data/knowledge/", "purpose": "Legacy crawled page JSON store", "classification": "REMOVE"},
    ],
    "B_html_ingestion": [
        {"path": "app/kb/services/kb_service.py", "purpose": "RAW artifact ingest (HTML/PDF from crawl era)", "classification": "REFACTOR"},
        {"path": "app/kb/services/document_builder.py", "purpose": "Markdown/HTML to structured content", "classification": "KEEP"},
        {"path": "app/kb/structuring/markdown.py", "purpose": "Markdown structure parser", "classification": "KEEP"},
        {"path": "app/kb/cleaning/generic.py", "purpose": "Generic HTML residue cleanup", "classification": "KEEP"},
        {"path": "app/kb/cleaning/earkart_in.py", "purpose": "Site-specific HTML cleaner", "classification": "REFACTOR"},
        {"path": "app/kb/cleaning/earkart_com.py", "purpose": "Shopify HTML cleaner", "classification": "REFACTOR"},
        {"path": "app/kb/cleaning/registry.py", "purpose": "Cleaner routing by website", "classification": "REFACTOR"},
        {"path": "data/raw/", "purpose": "Frozen immutable RAW web corpus", "classification": "KEEP"},
        {"path": "data/cleaned/", "purpose": "Frozen cleaned web artifacts", "classification": "KEEP"},
    ],
    "C_shopify": [
        {"path": "app/kb/cleaning/earkart_com.py", "purpose": "Shopify UI stripping", "classification": "REFACTOR"},
        {"path": "app/kb/chunking/noise.py", "purpose": "Shopify UI chunk detection", "classification": "REFACTOR"},
        {"path": "app/kb/retrieval/renderers/common.py", "purpose": "Shopify form sanitization", "classification": "REFACTOR"},
        {"path": "app/kb/audit/duplicates.py", "purpose": "Shopify boilerplate duplicate detection", "classification": "REFACTOR"},
    ],
    "D_crawler": [
        {"path": "app/helpers/crawler.py", "purpose": "Crawl4AI BFS deep crawl", "classification": "REMOVE"},
        {"path": "app/kb/policy/domain_policy.py", "purpose": "Crawl allowlist/exclusions", "classification": "REFACTOR"},
        {"path": "requirements.txt", "purpose": "crawl4ai dependency", "classification": "REFACTOR"},
    ],
    "E_raw_artifacts": [
        {"path": "app/kb/models/raw.py", "purpose": "RawArtifact domain models", "classification": "KEEP"},
        {"path": "app/kb/infrastructure/repositories/raw_repository.py", "purpose": "Append-only RAW storage", "classification": "KEEP"},
        {"path": "app/kb/infrastructure/repositories/cleaned_repository.py", "purpose": "Cleaned artifact persistence", "classification": "KEEP"},
        {"path": "app/kb/infrastructure/repositories/canonical_repository.py", "purpose": "Versioned canonical documents", "classification": "KEEP"},
        {"path": "app/kb/models/canonical.py", "purpose": "Canonical document entity", "classification": "KEEP"},
        {"path": "app/kb/services/cleaning_pipeline.py", "purpose": "RAW to cleaned to canonical", "classification": "KEEP"},
        {"path": "data/canonical/", "purpose": "Frozen canonical KB (V1 experiment)", "classification": "KEEP"},
    ],
    "F_renderers": [
        {"path": "app/kb/retrieval/renderers/", "purpose": "Structured content to retrieval text", "classification": "KEEP"},
    ],
    "G_document_extraction": [
        {"path": "app/providers/pdf/pdf_extractor.py", "purpose": "PDF native + OCR extraction", "classification": "KEEP"},
        {"path": "app/providers/pdf/pdf_layout.py", "purpose": "Page layout/block extraction", "classification": "KEEP"},
        {"path": "app/providers/pdf/pdf_quality.py", "purpose": "Extraction quality scoring", "classification": "KEEP"},
        {"path": "app/kb/structuring/pdf.py", "purpose": "PDF pages to DocumentContent", "classification": "KEEP"},
        {"path": "app/kb/cleaning/pdf.py", "purpose": "PDF text de-noising", "classification": "KEEP"},
        {"path": "app/kb/ingestion/", "purpose": "Phase 12 document-first ingestion (NEW)", "classification": "REPLACE"},
    ],
    "H_structure_recovery": [
        {"path": "app/kb/retrieval/recovery.py", "purpose": "Heading/list/section recovery", "classification": "KEEP"},
        {"path": "app/kb/models/structured_content.py", "purpose": "Content node tree models", "classification": "KEEP"},
        {"path": "app/kb/retrieval/classifier.py", "purpose": "Document type classification", "classification": "KEEP"},
    ],
    "I_chunking": [
        {"path": "app/kb/chunking/engine.py", "purpose": "Hierarchical semantic chunking", "classification": "KEEP"},
        {"path": "app/kb/chunking/config.py", "purpose": "512-token ceiling, overlap rules", "classification": "KEEP"},
        {"path": "app/kb/chunking/semantic_units.py", "purpose": "Semantic unit builder", "classification": "KEEP"},
        {"path": "app/kb/chunking/noise.py", "purpose": "Noise suppression", "classification": "KEEP"},
        {"path": "app/kb/chunking/gate.py", "purpose": "Production embedding gate", "classification": "KEEP"},
        {"path": "data/chunks/2026-08-17-v1/", "purpose": "Frozen V1 production chunks (5904)", "classification": "KEEP"},
    ],
    "J_embedding": [
        {"path": "app/kb/embedding/", "purpose": "Qwen provider + pipeline + versioning", "classification": "KEEP"},
        {"path": "app/kb/embedding/versioning.py", "purpose": "VectorIndexIdentity", "classification": "KEEP"},
    ],
    "K_pgvector": [
        {"path": "app/kb/vector/store.py", "purpose": "PGVector upsert/search", "classification": "KEEP"},
        {"path": "app/kb/vector/schema.py", "purpose": "V1 chunk_embeddings table", "classification": "KEEP"},
        {"path": "app/kb/vector/phase12_schema.py", "purpose": "Phase 12 versioned vector table (NEW)", "classification": "REPLACE"},
    ],
    "L_retrieval": [
        {"path": "app/kb/retrieval/service.py", "purpose": "Retrieval content preparation", "classification": "KEEP"},
        {"path": "app/kb/retrieval/models.py", "purpose": "RetrievalDocument models", "classification": "KEEP"},
        {"path": "app/kb/vector/search.py", "purpose": "VectorSearchService", "classification": "KEEP"},
    ],
    "M_evaluation": [
        {"path": "app/kb/evaluation/", "purpose": "Retrieval evaluation + diagnostics", "classification": "KEEP"},
    ],
    "N_provenance": [
        {"path": "app/kb/integrity/", "purpose": "KB freeze, manifest, baseline", "classification": "KEEP"},
        {"path": "app/kb/hashing.py", "purpose": "SHA-256 content hashing", "classification": "KEEP"},
    ],
    "O_scraping_tests": [
        {"path": "tests/test_kb_service.py", "purpose": "ingest_html end-to-end", "classification": "REFACTOR"},
        {"path": "tests/test_cleaning.py", "purpose": "HTML cleaning fixtures", "classification": "KEEP"},
        {"path": "tests/test_cleaning_pipeline.py", "purpose": "Pipeline on HTML fixtures", "classification": "KEEP"},
        {"path": "tests/test_legacy_importer.py", "purpose": "Legacy import plan", "classification": "REFACTOR"},
        {"path": "tests/fixtures/cleaning/", "purpose": "Frozen HTML regression fixtures", "classification": "KEEP"},
    ],
}
# fmt: on


def _summarize() -> dict[str, int]:
    counts: dict[str, int] = {}
    for items in INVENTORY.values():
        for item in items:
            label = item["classification"]
            counts[label] = counts.get(label, 0) + 1
    return counts


def _format_text(report: dict) -> str:
    lines = [
        "Phase 12 Architecture Audit",
        f"Generated: {report['generated_at']}",
        "",
        "SUMMARY",
        "-------",
    ]
    for label, count in sorted(report["summary"].items()):
        lines.append(f"{label}: {count}")
    lines.extend(
        [
            "",
            "MIGRATION STRATEGY",
            "------------------",
            "1. REPLACE web crawl ingestion with document-first upload pipeline (PDF/DOC/DOCX).",
            "2. REMOVE legacy crawl routes, Crawl4AI, and data/knowledge/ keyword chat path.",
            "3. KEEP frozen V1 experiment: data/chunks/2026-08-17-v1 + chunk_embeddings (5904 vectors).",
            "4. KEEP generic chunking, embedding, PGVector, evaluation, provenance infrastructure.",
            "5. REFACTOR site-specific Shopify/HTML cleaners into optional profiles (not production path).",
            "",
        ]
    )
    for category, items in report["categories"].items():
        lines.append(category.upper().replace("_", " "))
        lines.append("-" * len(category))
        for item in items:
            lines.append(f"  [{item['classification']}] {item['path']}")
            lines.append(f"      {item['purpose']}")
        lines.append("")
    lines.extend(
        [
            "V1 EXPERIMENT (UNTOUCHED)",
            "-------------------------",
            "data/chunks/2026-08-17-v1/ - 5904 frozen chunks",
            "PGVector table chunk_embeddings - embedding_version qwen_Qwen3-Embedding-0.6B_v1",
            "",
            "PHASE 12 TARGET",
            "---------------",
            "PDF / DOC / DOCX -> Upload -> Validate -> Extract -> OCR (selective) ->",
            "Structure -> Chunk -> Quality Gate -> Qwen Embed -> versioned PGVector",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    from app.config import settings

    summary = _summarize()
    report = {
        "report_version": "1.0",
        "read_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "v1_experiment": {
            "chunks_dir": "data/chunks/2026-08-17-v1",
            "chunk_count": 5904,
            "vector_table": settings.vector_table,
            "embedding_version": settings.embedding_version,
            "kb_dataset_version": settings.kb_dataset_version,
            "policy": "DO NOT MODIFY",
        },
        "phase12_target": {
            "input_types": ["pdf", "doc", "docx"],
            "documents_dir": "data/documents/",
            "vector_table": settings.phase12_vector_table,
            "embedding_version": settings.phase12_embedding_version,
            "kb_dataset_version": settings.phase12_kb_dataset_version,
        },
        "categories": INVENTORY,
    }

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase12_architecture_audit.json"
    txt_path = reports_dir / "phase12_architecture_audit.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(_format_text(report), encoding="utf-8")
    print(_format_text(report))
    print(f"\nWrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
