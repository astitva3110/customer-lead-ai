from __future__ import annotations

import json

from app.helpers.generation_json import extract_json_object, strip_markdown_fence


def test_extracts_plain_json_object() -> None:
    payload = extract_json_object('{"grounded": true, "answer": "ok", "source_ids": ["c1"]}')
    assert payload == {"grounded": True, "answer": "ok", "source_ids": ["c1"]}


def test_extracts_markdown_fenced_json() -> None:
    raw = """```json
{"grounded": true, "answer": "...", "source_ids": ["chunk_1"]}
```"""
    payload = extract_json_object(raw)
    assert payload is not None
    assert payload["grounded"] is True
    assert payload["source_ids"] == ["chunk_1"]


def test_extracts_json_embedded_in_prose() -> None:
    raw = 'Here you go:\n{"grounded": false, "answer": "no", "source_ids": []}\nthanks'
    payload = extract_json_object(raw)
    assert payload is not None
    assert payload["grounded"] is False


def test_plain_text_is_not_json() -> None:
    assert extract_json_object("FAME costs ₹10,999.") is None


def test_strip_fence_without_language() -> None:
    assert json.loads(strip_markdown_fence('```\n{"a": 1}\n```')) == {"a": 1}
