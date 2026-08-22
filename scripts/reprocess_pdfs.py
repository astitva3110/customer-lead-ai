"""Reprocess PDFs from RAW URLs with improved extraction (RAW artifacts remain immutable)."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


MEANINGFUL_CHAR_RE = re.compile(r"[\w]", re.UNICODE)


@dataclass
class ReprocessRecord:
    url: str
    website: str
    old_chars: int
    new_chars: int
    old_method: str
    new_method: str
    status: str
    error: str | None = None


@dataclass
class ReprocessReport:
    total: int = 0
    improved: int = 0
    unchanged: int = 0
    failed: int = 0
    substantially_improved: int = 0
    regressions: int = 0
    records: list[ReprocessRecord] = field(default_factory=list)


def _meaningful_chars(text: str) -> int:
    return len(MEANINGFUL_CHAR_RE.findall(text))


def _strip_wrapper(text: str) -> str:
    return re.sub(r"^#\s+.+\n\nSource:\s+https?://[^\n]+\n\n", "", text.strip(), count=1)


def main() -> int:
    from app.config import settings
    from app.kb.enums import SourceType
    from app.kb.services.kb_service import KnowledgeBaseService
    from app.providers.pdf.pdf_extractor import extract_pdf

    kb = KnowledgeBaseService()
    report = ReprocessReport()

    seen_urls: set[str] = set()
    with httpx.Client(headers={"User-Agent": "earkart-chatbot/1.0"}, timeout=120) as client:
        for website_dir in sorted(kb.raw_store.base_dir.iterdir()):
            if not website_dir.is_dir():
                continue
            website = website_dir.name
            index = kb.raw_store._load_index(website)
            for canonical_url, entry in index.get("documents", {}).items():
                source_type = entry.get("source_type")
                if source_type not in (SourceType.PDF.value, SourceType.SCANNED_PDF.value):
                    continue
                if canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)

                old_artifact = kb.raw_store.get_latest_for_url(website, canonical_url)
                if old_artifact is None:
                    continue

                old_text = _strip_wrapper(old_artifact.content)
                old_chars = _meaningful_chars(old_text)
                report.total += 1

                try:
                    result = extract_pdf(canonical_url, client, ocr_fallback=True)
                except Exception as exc:
                    report.failed += 1
                    report.records.append(
                        ReprocessRecord(
                            url=canonical_url,
                            website=website,
                            old_chars=old_chars,
                            new_chars=0,
                            old_method=old_artifact.extraction_method.value,
                            new_method="",
                            status="failed",
                            error=str(exc),
                        )
                    )
                    continue

                new_chars = _meaningful_chars(result.full_text)
                new_method = result.extraction_method.value
                delta = new_chars - old_chars

                if not result.full_text.strip():
                    status = "failed"
                    report.failed += 1
                elif delta > 50:
                    status = "substantially_improved"
                    report.substantially_improved += 1
                    report.improved += 1
                elif delta > 0:
                    status = "improved"
                    report.improved += 1
                elif delta < -50:
                    status = "regression"
                    report.regressions += 1
                else:
                    status = "unchanged"
                    report.unchanged += 1

                report.records.append(
                    ReprocessRecord(
                        url=canonical_url,
                        website=website,
                        old_chars=old_chars,
                        new_chars=new_chars,
                        old_method=old_artifact.extraction_method.value,
                        new_method=new_method,
                        status=status,
                    )
                )

                if status in ("improved", "substantially_improved") and new_chars != old_chars:
                    kb.ingest_pdf(
                        canonical_url,
                        result,
                        crawl_root_url=old_artifact.crawl_root_url,
                        scraped_at=datetime.now(timezone.utc),
                    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": report.total,
            "improved": report.improved,
            "unchanged": report.unchanged,
            "failed": report.failed,
            "substantially_improved": report.substantially_improved,
            "regressions": report.regressions,
        },
        "records": [record.__dict__ for record in report.records],
        "radius_pdfs": [
            record.__dict__
            for record in report.records
            if "radius" in record.url.lower()
        ],
    }
    out_path = settings.reports_dir / "pdf_reprocess_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total PDFs: {report.total}")
    print(f"Improved: {report.improved} (substantially: {report.substantially_improved})")
    print(f"Unchanged: {report.unchanged}")
    print(f"Failed: {report.failed}")
    print(f"Regressions: {report.regressions}")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
