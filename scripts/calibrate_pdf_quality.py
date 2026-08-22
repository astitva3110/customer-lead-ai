"""Calibrate PDF extraction quality thresholds from existing RAW data."""
import json
import re
from pathlib import Path

stats: dict[str, list] = {"pdf": [], "scanned_pdf": []}
for website_dir in Path("data/raw").iterdir():
    if not website_dir.is_dir():
        continue
    for st in ("pdf", "scanned_pdf"):
        d = website_dir / st
        if not d.exists():
            continue
        for f in d.glob("sha256_*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            content = data.get("content", "")
            text = re.sub(r"^#\s+.+\n\nSource:\s+https?://[^\n]+\n\n", "", content.strip(), count=1)
            chars = len(re.findall(r"[\w]", text, re.UNICODE))
            words = len(text.split())
            lines = len([line for line in text.split("\n") if line.strip()])
            stats[st].append(
                {
                    "url": data.get("url", ""),
                    "chars": chars,
                    "words": words,
                    "lines": lines,
                    "method": data.get("extraction_method"),
                }
            )

for st, items in stats.items():
    if not items:
        continue
    chars = sorted(i["chars"] for i in items)
    words = sorted(i["words"] for i in items)
    n = len(chars)
    print(f"=== {st} ({n} docs) ===")
    print(
        f"  chars: min={chars[0]}, p10={chars[n // 10]}, median={chars[n // 2]}, "
        f"p90={chars[9 * n // 10]}, max={chars[-1]}"
    )
    print(
        f"  words: min={words[0]}, p10={words[len(words) // 10]}, median={words[len(words) // 2]}, "
        f"p90={words[9 * len(words) // 10]}, max={words[-1]}"
    )
    sparse = [i for i in items if i["chars"] < 150]
    print(f"  sparse (<150 chars): {len(sparse)}")
    for s in sparse[:10]:
        print(f"    {s['chars']} chars: {s['url'][:80]}")
