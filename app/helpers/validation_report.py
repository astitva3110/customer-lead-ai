from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_txt(path_stem: Path, payload: dict[str, Any], text: str) -> tuple[Path, Path]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = path_stem.with_suffix(".json")
    txt_path = path_stem.with_suffix(".txt")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    txt_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return json_path, txt_path


def format_kv_report(title: str, payload: dict[str, Any], *, sections: list[tuple[str, Any]] | None = None) -> str:
    lines = [title, "=" * len(title), ""]
    for key, value in (sections or list(payload.items())):
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for nested_key, nested_value in value.items():
                lines.append(f"  {nested_key}: {nested_value}")
            lines.append("")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
            lines.append("")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines).rstrip() + "\n"
