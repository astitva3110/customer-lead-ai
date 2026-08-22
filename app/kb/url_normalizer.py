from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

# Shopify and common marketing/tracking params stripped for document identity.
TRACKING_QUERY_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "pr_prod_strat",
        "pr_rec_id",
        "pr_rec_pid",
        "pr_ref_pid",
        "pr_seq",
        "_ga",
        "_gl",
        "ref",
        "source",
    }
)

INDEX_ALIASES = frozenset({"index.html", "index.htm", "index.php", "default.html", "default.htm"})


def extract_website(url: str) -> str:
    """Return normalized website hostname (lowercase, no www.)."""
    parsed = urlparse(url.strip())
    host = parsed.netloc or parsed.path.split("/")[0]
    return host.lower().removeprefix("www.")


def normalize_url(url: str) -> str:
    """
    Normalize a URL for stable document identity.

    Rules:
    - lowercase hostname
    - decode percent-encoding in path
    - collapse /index.html to /
    - strip trailing slash (except root)
    - remove tracking query parameters
    - sort remaining query params
    """
    url = url.strip()
    if not url:
        return url

    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = parsed.netloc.lower().removeprefix("www.")

    path = unquote(parsed.path or "")
    if not path:
        path = "/"
    elif path != "/":
        segments = path.rstrip("/").split("/")
        if segments and segments[-1].lower() in INDEX_ALIASES:
            segments = segments[:-1]
            path = "/" + "/".join(s for s in segments if s)
            if not path or path == "":
                path = "/"
        path = path.rstrip("/") if path != "/" else path

    query_params = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {
        key: values
        for key, values in query_params.items()
        if key.lower() not in TRACKING_QUERY_PARAMS
    }
    query = urlencode(sorted(filtered.items()), doseq=True)

    return urlunparse((scheme, host, path, "", query, ""))


def document_identity_key(website: str, url: str) -> str:
    """Stable identity key: website + normalized URL."""
    return f"{website.lower()}:{normalize_url(url)}"


def make_document_id(website: str, url: str) -> str:
    """Deterministic UUID5 document id from website + canonical URL."""
    import uuid

    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    key = document_identity_key(website, url)
    return str(uuid.uuid5(namespace, key))
