import os

# Keep the full pytest suite from sharing one in-memory limiter bucket.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
