import json
from pathlib import Path

from app.kb.services.legacy_importer import legacy_record_to_raw, plan_legacy_import


def test_legacy_import_plan(tmp_path: Path) -> None:
    legacy = tmp_path / "earkart.in"
    legacy.mkdir(parents=True)
    (legacy / "0000.json").write_text(
        json.dumps(
            {
                "url": "https://earkart.in/about-us.html",
                "title": "About",
                "markdown": "# About",
                "source_url": "https://earkart.in/",
                "content_type": "html",
            }
        ),
        encoding="utf-8",
    )

    plans = plan_legacy_import(tmp_path)
    assert len(plans) == 1
    assert plans[0].website == "earkart.in"
    assert plans[0].extraction_method.value == "html_parser"


def test_legacy_ocr_inference() -> None:
    data = {
        "url": "https://earkart.in/investor/policy.pdf",
        "title": "[PDF [OCR]] policy.pdf",
        "markdown": "text",
        "content_type": "pdf",
    }
    artifact = legacy_record_to_raw(data)
    assert artifact.source_type.value == "scanned_pdf"
    assert artifact.extraction_method.value == "ocr"
