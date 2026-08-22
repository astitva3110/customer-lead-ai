import json
from pathlib import Path

report = json.loads(Path("reports/pdf_reprocess_report.json").read_text(encoding="utf-8"))
records = report["records"]

unchanged_sparse = [r for r in records if r["status"] == "unchanged" and r["new_chars"] < 150]
unchanged_low = [r for r in records if r["status"] == "unchanged" and 150 <= r["new_chars"] < 300]

print("=== UNCHANGED + SPARSE (<150 meaningful chars) ===")
for r in sorted(unchanged_sparse, key=lambda x: x["new_chars"]):
    method = r["new_method"] or r["old_method"]
    print(f"  {r['new_chars']:4d} chars [{method:8s}] {r['url']}")
print(f"\nTotal: {len(unchanged_sparse)}")

print("\n=== UNCHANGED + LOW (150-299 meaningful chars) ===")
for r in sorted(unchanged_low, key=lambda x: x["new_chars"]):
    method = r["new_method"] or r["old_method"]
    print(f"  {r['new_chars']:4d} chars [{method:8s}] {r['url']}")
print("\n=== LOWEST UNCHANGED PDFs (still minimal relative to dataset) ===")
unchanged = [r for r in records if r["status"] == "unchanged"]
unchanged.sort(key=lambda x: x["new_chars"])
for r in unchanged[:25]:
    method = r["new_method"] or r["old_method"]
    print(f"  {r['new_chars']:5d} chars [{method:8s}] {r['url']}")

print("\n=== FAILED ===")
for r in records:
    if r["status"] == "failed":
        print(f"  {r['url']} — {r.get('error')}")

