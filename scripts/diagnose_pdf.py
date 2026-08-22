"""Temporary diagnostic script for PDF extraction."""
import httpx
import pymupdf

url = "https://earkart.in/radius/RADIUSM16BTE.pdf"
r = httpx.get(url, follow_redirects=True, timeout=60)
doc = pymupdf.open(stream=r.content, filetype="pdf")
print("Pages:", doc.page_count)
print("Metadata:", doc.metadata)
for i, page in enumerate(doc, 1):
    print(f"\n=== Page {i} ===")
    text = page.get_text()
    print("get_text() length:", len(text))
    print("get_text() preview:", repr(text[:800]))
    blocks = page.get_text("blocks")
    print("blocks count:", len(blocks))
    for j, b in enumerate(blocks[:15]):
        print(f"  block {j}: {repr(b)}")
    words = page.get_text("words")
    print("words count:", len(words))
    if words:
        print("first 15 words:", words[:15])
    dict_text = page.get_text("dict")
    print("dict blocks:", len(dict_text.get("blocks", [])))
    for j, block in enumerate(dict_text.get("blocks", [])):
        btype = block.get("type")
        if btype == 0:
            for line in block.get("lines", []):
                spans = [s.get("text", "") for s in line.get("spans", [])]
                print(f"  text block {j}: {spans}")
        elif btype == 1:
            print(f"  image block {j}: bbox={block.get('bbox')}")
    imgs = page.get_images()
    print("images count:", len(imgs))
    for img in imgs[:5]:
        print(f"  image: {img}")
doc.close()
