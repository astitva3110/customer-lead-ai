"""Canonical KB freeze and integrity orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.kb.audit.collect import collect_document_bundles, iter_canonical_documents
from app.kb.enums import ExclusionReason, ProcessingStatus
from app.kb.integrity.checker import DocumentIntegrityResult, check_document_integrity
from app.kb.integrity.contract import DownstreamDocument, downstream_eligible
from app.kb.integrity.exclusion import check_exclusion_integrity
from app.kb.integrity.failed_pdfs import assess_all_failed_pdfs
from app.kb.integrity.identity import check_identity_integrity
from app.kb.integrity.manifest import build_manifest
from app.kb.integrity.version import KB_DATASET_VERSION
from app.kb.models.canonical import CanonicalDocument, utc_now
from app.kb.policy.domain_policy import build_exclusion_metadata
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore


@dataclass
class FreezeResult:
    failed_pdf_assessments: list[dict[str, Any]]
    excluded_failed_count: int
    document_results: list[DocumentIntegrityResult] = field(default_factory=list)
    identity_report: dict[str, Any] = field(default_factory=dict)
    exclusion_report: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    integrity_report: dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_dataset_version": KB_DATASET_VERSION,
            "passed": self.passed,
            "failed_pdf_assessments": self.failed_pdf_assessments,
            "excluded_failed_count": self.excluded_failed_count,
            "identity_report": self.identity_report,
            "exclusion_report": self.exclusion_report,
            "manifest_summary": {
                k: self.manifest.get(k)
                for k in (
                    "generated_at",
                    "kb_dataset_version",
                    "total_documents",
                    "active_count",
                    "validated_count",
                    "excluded_count",
                    "failed_count",
                    "production_count",
                )
            },
            "integrity_report": self.integrity_report,
        }


class CanonicalKbFreezer:
    """Freeze canonical KB: resolve FAILED PDFs, verify integrity, emit manifest."""

    def __init__(
        self,
        raw_store: RawStore,
        cleaned_store: CleanedStore,
        canonical_store: CanonicalStore,
        reports_dir: Path,
    ) -> None:
        self.raw_store = raw_store
        self.cleaned_store = cleaned_store
        self.canonical_store = canonical_store
        self.reports_dir = reports_dir

    def _apply_exclusion(self, document: CanonicalDocument, reason: ExclusionReason) -> CanonicalDocument:
        updated = document.model_copy(
            update={
                "processing_status": ProcessingStatus.EXCLUDED,
                "updated_at": utc_now(),
                "metadata": {
                    **document.metadata,
                    **build_exclusion_metadata(document.website, reason),
                    "freeze_note": "Excluded during Phase 5 canonical freeze",
                    "previous_status": document.processing_status.value,
                },
            }
        )
        self.canonical_store.write(updated)
        return updated

    def resolve_failed_pdfs(self, bundles) -> tuple[list[dict], int]:
        assessments = assess_all_failed_pdfs(bundles)
        excluded = 0
        records = []
        for assessment in assessments:
            record = assessment.to_dict()
            if assessment.classification == "UNRECOVERABLE":
                for bundle in bundles:
                    if bundle.canonical.document_id == assessment.document_id:
                        self._apply_exclusion(bundle.canonical, ExclusionReason.EXTRACTION_FAILED)
                        excluded += 1
                        record["action_taken"] = "EXCLUDED with extraction_failed"
                        break
            else:
                record["action_taken"] = "none — requires manual reprocessing"
            records.append(record)
        return records, excluded

    def run(self) -> FreezeResult:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        bundles = collect_document_bundles(self.raw_store, self.cleaned_store, self.canonical_store)
        failed_assessments, excluded_count = self.resolve_failed_pdfs(bundles)

        assessments_path = self.reports_dir / "phase5_failed_pdf_analysis.json"
        assessments_path.write_text(
            json.dumps(
                {
                    "generated_at": utc_now().isoformat(),
                    "assessments": failed_assessments,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        documents = iter_canonical_documents(self.canonical_store)
        doc_results = [check_document_integrity(doc) for doc in documents]
        identity = check_identity_integrity(documents)
        exclusion = check_exclusion_integrity(documents)
        manifest = build_manifest(documents)

        production_results = [
            r
            for r in doc_results
            if r.processing_status in ("ACTIVE", "VALIDATED")
        ]
        production_passed = all(r.passed for r in production_results)

        failed_remaining = [r for r in doc_results if r.processing_status == "FAILED"]
        excluded_ok = exclusion.passed
        identity_ok = identity.passed
        no_failed = len(failed_remaining) == 0

        integrity_report = {
            "kb_dataset_version": KB_DATASET_VERSION,
            "total_canonical_documents": len(documents),
            "active_count": manifest["active_count"],
            "validated_count": manifest["validated_count"],
            "excluded_count": manifest["excluded_count"],
            "failed_count": manifest["failed_count"],
            "production_count": manifest["production_count"],
            "production_integrity_passed": production_passed,
            "identity_integrity_passed": identity_ok,
            "exclusion_integrity_passed": excluded_ok,
            "failed_documents_remaining": len(failed_remaining),
            "duplicate_document_ids": len(identity.duplicate_document_ids),
            "duplicate_canonical_urls": len(identity.duplicate_canonical_urls),
            "document_integrity_failures": [
                r.to_dict() for r in doc_results if not r.passed and r.processing_status in ("ACTIVE", "VALIDATED")
            ],
            "failed_documents": [r.to_dict() for r in failed_remaining],
            "content_consistency_problems": sum(
                1
                for r in doc_results
                for i in r.issues
                if i.category == "consistency" and i.severity == "ERROR"
            ),
            "downstream_eligible_count": sum(1 for d in documents if downstream_eligible(d)),
        }

        passed = production_passed and identity_ok and excluded_ok and no_failed

        manifest_path = self.reports_dir / "canonical_kb_manifest.json"
        integrity_path = self.reports_dir / "canonical_kb_integrity.json"
        summary_path = self.reports_dir / "canonical_kb_summary.txt"
        phase5_report_path = self.reports_dir / "phase5_integrity_report.json"
        phase5_summary_path = self.reports_dir / "phase5_summary.txt"

        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        integrity_path.write_text(
            json.dumps(
                {
                    "identity": identity.to_dict(),
                    "exclusion": exclusion.to_dict(),
                    "documents": [r.to_dict() for r in doc_results],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        phase5_report_path.write_text(json.dumps(integrity_report, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_lines = [
            "Canonical Knowledge Base — Frozen Snapshot",
            f"KB Dataset Version: {KB_DATASET_VERSION}",
            f"Generated: {manifest['generated_at']}",
            "",
            f"Total documents: {manifest['total_documents']}",
            f"Production (ACTIVE + VALIDATED): {manifest['production_count']}",
            f"  ACTIVE: {manifest['active_count']}",
            f"  VALIDATED: {manifest['validated_count']}",
            f"EXCLUDED: {manifest['excluded_count']}",
            f"FAILED: {manifest['failed_count']}",
            "",
            f"Downstream eligible: {integrity_report['downstream_eligible_count']}",
            f"Production integrity: {'PASSED' if production_passed else 'FAILED'}",
            f"Identity integrity: {'PASSED' if identity_ok else 'FAILED'}",
            f"Exclusion integrity: {'PASSED' if excluded_ok else 'FAILED'}",
            f"Freeze acceptance: {'PASSED' if passed else 'FAILED'}",
        ]
        summary_text = "\n".join(summary_lines) + "\n"
        summary_path.write_text(summary_text, encoding="utf-8")
        phase5_summary_path.write_text(summary_text, encoding="utf-8")

        return FreezeResult(
            failed_pdf_assessments=failed_assessments,
            excluded_failed_count=excluded_count,
            document_results=doc_results,
            identity_report=identity.to_dict(),
            exclusion_report=exclusion.to_dict(),
            manifest=manifest,
            integrity_report=integrity_report,
            passed=passed,
        )
