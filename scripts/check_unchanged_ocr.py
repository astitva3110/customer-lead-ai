import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import pymupdf

report = json.loads(Path("reports/pdf_reprocess_report.json").read_text(encoding="utf-8"))
unchanged_ocr = [r for r in report["records"] if r["status"] == "unchanged" and r["new_method"] == "ocr"]
unchanged_ocr.sort(key=lambda x: x["new_chars"])

client = httpx.Client(timeout=60, headers={"User-Agent": "earkart-chatbot/1.0"})

print("=== UNCHANGED OCR PDFs with lowest content ===")
for r in unchanged_ocr[:20]:
    url = r["url"]
    try:
        resp = client.get(url)
        doc = pymupdf.open(stream=resp.content, filetype="pdf")
        pages = doc.page_count
        doc.close()
        per_page = r["new_chars"] / max(pages, 1)
        flag = " LOW" if per_page < 100 and pages > 1 else ""
        print(f"  {r['new_chars']:5d} chars, {pages:3d} pages ({per_page:.0f}/page){flag}  {url.split('/')[-1]}")
    except Exception as e:
        print(f"  {r['new_chars']:5d} chars  ERROR  {url}")

client.close()

print("\n=== SUMMARY: unchanged but still suspiciously sparse (<150 chars) ===")
sparse = [r for r in report["records"] if r["status"] == "unchanged" and r["new_chars"] < 150]
print(f"Count: {len(sparse)}")
for r in sparse:
    print(f"  {r['url']}")
