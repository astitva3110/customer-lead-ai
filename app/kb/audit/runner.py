from pathlib import Path

from app.config import settings
from app.kb.audit.collect import (
    collect_document_bundles,
    iter_cleaned_artifacts,
    iter_raw_artifacts,
)
from app.kb.audit.document_audit import audit_document
from app.kb.audit.duplicates import run_duplicate_analysis
from app.kb.audit.models import ReviewItem, ReviewSeverity
from app.kb.audit.reporter import build_summary, write_reports
from app.kb.services.kb_service import KnowledgeBaseService
from app.kb.services.legacy_importer import import_legacy_directory
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore


class KnowledgeBaseAuditor:
    """Process RAW artifacts and produce quality audit reports."""

    def __init__(
        self,
        raw_store: RawStore | None = None,
        cleaned_store: CleanedStore | None = None,
        canonical_store: CanonicalStore | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self.raw_store = raw_store or RawStore(settings.raw_dir)
        self.cleaned_store = cleaned_store or CleanedStore(settings.cleaned_dir)
        self.canonical_store = canonical_store or CanonicalStore(settings.canonical_dir)
        self.reports_dir = reports_dir or settings.reports_dir
        self.kb_service = KnowledgeBaseService(
            raw_store=self.raw_store,
            cleaned_store=self.cleaned_store,
            canonical_store=self.canonical_store,
        )

    def import_legacy(self, legacy_dir: Path | None = None, website: str | None = None) -> int:
        legacy_path = legacy_dir or settings.knowledge_dir
        imported = import_legacy_directory(
            legacy_path,
            self.raw_store,
            website=website,
            dry_run=False,
        )
        return len(imported)

    def process_raw(self, website: str | None = None) -> int:
        processed = 0
        for path in sorted(self.raw_store.base_dir.rglob("sha256_*.json")):
            artifact = self.raw_store.read_path(path)
            if website and artifact.website != website:
                continue
            self.kb_service.process_raw_artifact(artifact)
            processed += 1
        return processed

    def run_audit(self) -> dict:
        bundles = collect_document_bundles(self.raw_store, self.cleaned_store, self.canonical_store)
        records = [audit_document(bundle) for bundle in bundles]
        duplicate_report = run_duplicate_analysis(bundles)

        review_items: list[ReviewItem] = []
        for record in records:
            review_items.extend(record.review_items)

        for section, groups in duplicate_report.items():
            for group in groups:
                for doc in group.get("documents", []):
                    review_items.append(
                        ReviewItem(
                            document_id=doc.get("document_id", ""),
                            website=doc.get("website", ""),
                            url=doc.get("url") or doc.get("canonical_url") or doc.get("source_url", ""),
                            title=doc.get("title", ""),
                            source_type="",
                            issue=f"duplicate detected: {group.get('type', section)}",
                            severity=ReviewSeverity.INFO,
                            context={"group": group.get("type", section)},
                        )
                    )

        raw_count = len(iter_raw_artifacts(self.raw_store))
        cleaned_count = len(iter_cleaned_artifacts(self.cleaned_store))
        summary = build_summary(
            raw_count=raw_count,
            cleaned_count=cleaned_count,
            records=records,
            duplicate_report=duplicate_report,
        )
        report_paths = write_reports(self.reports_dir, summary, records, duplicate_report, review_items)

        return {
            "summary": summary,
            "records": records,
            "duplicate_report": duplicate_report,
            "review_items": review_items,
            "report_paths": report_paths,
        }


def run_kb_audit(
    *,
    import_legacy: bool = False,
    process: bool = True,
    website: str | None = None,
    legacy_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    auditor = KnowledgeBaseAuditor(reports_dir=reports_dir)
    imported = 0
    processed = 0

    if import_legacy:
        imported = auditor.import_legacy(legacy_dir=legacy_dir, website=website)

    if process:
        processed = auditor.process_raw(website=website)

    result = auditor.run_audit()
    result["imported"] = imported
    result["processed"] = processed
    return result
