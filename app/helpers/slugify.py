import re
from urllib.parse import urlparse


def slugify_url(url: str) -> str:
    host = urlparse(url).netloc.replace("www.", "")
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", host).strip("-")
    return safe or "crawl"
