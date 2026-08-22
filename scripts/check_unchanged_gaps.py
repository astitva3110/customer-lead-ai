"""Check unchanged PDFs that may still have poor extraction vs OCR potential."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.providers.pdf.pdf_extractor import extract_pdf_bytes
from app.providers.pdf.pdf_quality import meaningful_char_count, is_poor_quality, assess_extraction_quality
import pymupdf

report = json.loads(Path("reports/pdf_reprocess_report.json").read_text(encoding="utf-8"))
unchanged = [r for r in report["records"] if r["status"] == "unchanged"]
unchanged.sort(key=lambda x: x["new_chars"])

# Check lowest unchanged: would OCR get significantly more?
candidates = unchanged[:15]
client = httpx.Client(timeout=60, headers={"User-Agent": "earkart-chatbot/1.0"})

print("Checking if unchanged low-content PDFs would benefit from OCR:\n")
for r in candidates:
    url = r["url"]
    try:
        resp = client.get(url)
        native = extract_pdf_bytes(resp.content, ocr_fallback=False)
        full = extract_pdf_bytes(resp.content, ocr_fallback=True)
        native_chars = meaningful_char_count(native.full_text)
        full_chars = meaningful_char_count(full.full_text)
        delta = full_chars - native_chars
        flag = " *** POSSIBLE GAP ***" if delta > 100 and r["new_method"] == "pdf_text" else ""
        print(f"{native_chars:4d} native -> {full_chars:4d} with OCR (delta +{delta}) [{r['new_method']}]{flag}")
        print(f"       {url}\n")
    except Exception as e:
        print(f"ERROR {url}: {e}\n")

client.close()
