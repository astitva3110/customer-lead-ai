import os

# Keep the full pytest suite from sharing one in-memory limiter bucket.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CHAT_SERVICE_TOKEN", "test-chat-service-token")


def chat_service_headers() -> dict[str, str]:
    token = os.environ.get("CHAT_SERVICE_TOKEN", "test-chat-service-token").strip()
    return {"Authorization": f"Bearer {token}"}
