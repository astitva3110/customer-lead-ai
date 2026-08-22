from app.kb.hashing import content_hash, hash_filename, hash_raw_payload, sha256_hex


def test_sha256_hex_deterministic() -> None:
    assert sha256_hex("hello") == sha256_hex("hello")
    assert sha256_hex("hello") != sha256_hex("world")


def test_content_hash_prefix() -> None:
    result = content_hash("test body")
    assert result.startswith("sha256:")
    assert len(result) == 71


def test_hash_raw_payload_uses_content_field() -> None:
    payload = {"content": "body", "title": "ignored for hash"}
    h1 = hash_raw_payload(payload)
    h2 = content_hash("body")
    assert h1 == h2


def test_hash_filename() -> None:
    assert hash_filename("sha256:abcd") == "sha256_abcd.json"
