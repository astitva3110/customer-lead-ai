"""Investigate Phase 3 FAILED documents and write classification report."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.kb.audit.collect import collect_document_bundles
from app.kb.storage.canonical_store import CanonicalStore
from app.kb.storage.cleaned_store import CleanedStore
from app.kb.storage.raw_store import RawStore
from app.kb.validation.document_validator import _alnum_count, baseline_text


def classify(bundle) -> dict:
    c = bundle.canonical
    raw = bundle.raw
    cleaned = bundle.cleaned
    errors = c.metadata.get("validation", {}).get("errors", [])
    url = c.canonical_url.lower()
    title = c.title.lower()

    baseline = baseline_text(raw) if raw else ""
    base_a = _alnum_count(baseline) if raw else 0
    clean_a = _alnum_count(cleaned.content) if cleaned else 0
    ratio = clean_a / base_a if base_a else None

    failure_reason = "; ".join(errors) if errors else "unknown"
    classification = "F"
    recommended = "manual review"

    if "404" in title or "not found" in title:
        classification = "E"
        recommended = "EXCLUDE: 404 page"
    elif "/cart" in url or "shopping cart" in title:
        classification = "E"
        recommended = "EXCLUDE: cart/checkout page — not KB content"
    elif c.website in {"crm.earkart.in", "firecrawl.dev"}:
        classification = "E"
        recommended = f"EXCLUDE: non-target website ({c.website})"
    elif "/apps/" in url:
        classification = "E"
        recommended = "EXCLUDE: third-party app embed page"
    elif c.source_type.value == "html" and any(
        "excessive content loss" in e for e in errors
    ):
        if ratio and ratio < 0.25 and clean_a < 200:
            if "/blogs/news?page=" in url or "/collections/" in url:
                classification = "B"
                recommended = "VALID high reduction OR exclude listing page — review baseline"
            elif "/pages/contact" in url:
                classification = "D"
                recommended = "Fix validation baseline (nav-heavy contact page)"
            elif "/blogs/news/" in url:
                classification = "A"
                recommended = "Fix cleaner — blog body removed"
            else:
                classification = "D"
                recommended = "Fix validation baseline for earkart.com (cart/nav not in baseline)"
        elif "/pages/" in url and ratio and 0.45 <= ratio < 0.50:
            classification = "D"
            recommended = "Fix validation baseline — policy page legitimately ~48% vs nav-heavy raw"
        elif "/collections/" in url and ratio and ratio < 0.50:
            classification = "B"
            recommended = "Collection listing — valid high reduction; consider EXCLUDE or relaxed validation"
        else:
            classification = "D"
            recommended = "Fix validation baseline for earkart.com HTML"
    elif c.source_type.value == "pdf" and any(
        "excessive content loss" in e for e in errors
    ):
        raw_chars = len(raw.content) if raw else 0
        clean_chars = len(cleaned.content) if cleaned else 0
        if clean_a >= 80 and ratio and ratio < 0.55:
            if "radius" in url or "cdr" in title.lower():
                classification = "C"
                recommended = "Sparse vector PDF extraction — compare raw vs cleaned; fix PDF baseline"
            elif "investor" in url:
                classification = "C"
                recommended = "Sparse investor notice PDF — extraction quality or header removal"
            else:
                classification = "C"
                recommended = "Sparse PDF — use PDF-specific baseline (exclude repeated headers)"
        else:
            classification = "C"
            recommended = "PDF extraction/cleaning investigation"
    elif any("too short" in e for e in errors):
        classification = "E"
        recommended = "EXCLUDE or mark FAILED — insufficient content"

    return {
        "document_id": c.document_id,
        "website": c.website,
        "url": c.canonical_url,
        "title": c.title,
        "source_type": c.source_type.value,
        "extraction_method": c.extraction_method.value,
        "failure_reason": failure_reason,
        "classification": classification,
        "classification_label": {
            "A": "REAL CLEANING FAILURE",
            "B": "VALID HIGH CONTENT REDUCTION",
            "C": "EXTRACTION QUALITY PROBLEM",
            "D": "VALIDATION THRESHOLD PROBLEM",
            "E": "BAD SOURCE / 404 / EMPTY PAGE",
            "F": "OTHER",
        }[classification],
        "recommended_action": recommended,
        "metrics": {
            "raw_char_count": len(raw.content) if raw else 0,
            "cleaned_char_count": len(cleaned.content) if cleaned else 0,
            "baseline_alnum": base_a,
            "cleaned_alnum": clean_a,
            "retention_vs_baseline": round(ratio, 4) if ratio is not None else None,
        },
        "cleaned_preview": (cleaned.content if cleaned else "")[:300],
        "validation_errors": errors,
        "validation_warnings": c.metadata.get("validation", {}).get("warnings", []),
    }


def main() -> int:
    raw_store = RawStore(settings.raw_dir)
    cleaned_store = CleanedStore(settings.cleaned_dir)
    canonical_store = CanonicalStore(settings.canonical_dir)
    bundles = collect_document_bundles(raw_store, cleaned_store, canonical_store)
    failed = [b for b in bundles if b.canonical.processing_status.value == "FAILED"]
    records = [classify(b) for b in failed]

    counts: dict[str, int] = {}
    for r in records:
        counts[r["classification_label"]] = counts.get(r["classification_label"], 0) + 1

    output = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "failed_count": len(records),
        "classification_summary": counts,
        "documents": records,
    }

    out_path = settings.reports_dir / "phase4_failed_documents_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(records)} documents)")
    print("Classification:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
