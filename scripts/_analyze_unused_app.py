"""One-off: list unused app modules by import reachability."""
from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"


def main() -> None:
    module_to_file: dict[str, Path] = {}
    file_to_module: dict[Path, str] = {}
    for py in APP.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        mod = ".".join(py.relative_to(ROOT).with_suffix("").parts)
        module_to_file[mod] = py
        file_to_module[py] = mod

    def resolve_relative(module_file: Path, node_level: int, node_module: str | None) -> str | None:
        pkg_parts = file_to_module[module_file].split(".")
        if node_level > len(pkg_parts):
            return None
        base = pkg_parts[: len(pkg_parts) - node_level]
        if node_module:
            return ".".join(base + node_module.split("."))
        return ".".join(base)

    def imports_in_file(py: Path) -> set[str]:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            return set()
        out: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                if node.level:
                    resolved = resolve_relative(py, node.level, node.module)
                    if resolved:
                        out.add(resolved)
                        for alias in node.names:
                            out.add(f"{resolved}.{alias.name}")
                else:
                    out.add(node.module)
                    for alias in node.names:
                        out.add(f"{node.module}.{alias.name}")
        return out

    direct_deps: dict[str, set[str]] = {}
    for py, mod in file_to_module.items():
        deps = {imp for imp in imports_in_file(py) if imp.startswith("app")}
        direct_deps[mod] = deps

    def resolve_module(imp: str) -> str | None:
        if imp in module_to_file:
            return imp
        parts = imp.split(".")
        while parts:
            cand = ".".join(parts)
            if cand in module_to_file:
                return cand
            parts.pop()
        return None

    def reachable_from(seeds: list[str]) -> set[str]:
        seen: set[str] = set()
        q: deque[str] = deque()
        for seed in seeds:
            resolved = resolve_module(seed)
            if resolved:
                q.append(resolved)
        while q:
            mod = q.popleft()
            if mod in seen:
                continue
            seen.add(mod)
            for imp in direct_deps.get(mod, ()):
                target = resolve_module(imp)
                if target and target not in seen:
                    q.append(target)
        return seen

    runtime_seeds = [
        "app.main",
        "app.dependencies",
        "app.error_handlers",
        "app.db.engine",
        "app.db.migrations",
    ]
    runtime = reachable_from(runtime_seeds)

    external_seeds: set[str] = set()
    for search in (ROOT / "tests", ROOT / "scripts"):
        if not search.exists():
            continue
        for py in search.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            for imp in imports_in_file(py):
                if imp.startswith("app"):
                    resolved = resolve_module(imp)
                    if resolved:
                        external_seeds.add(resolved)
    repo_only = reachable_from(list(external_seeds)) - runtime

    all_mods = {
        m
        for m in file_to_module.values()
        if module_to_file[m].name != "__init__.py"
    }

    runtime_unused = sorted(all_mods - runtime)
    orphans = sorted(all_mods - reachable_from(runtime_seeds + list(external_seeds)))

    def group(mods: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for mod in mods:
            parts = mod.split(".")
            key = parts[1] if len(parts) > 1 else parts[0]
            grouped[key].append(mod)
        return grouped

    print("=== RUNTIME UNUSED (not reachable from main/dependencies) ===")
    print(f"Count: {len(runtime_unused)} / {len(all_mods)}")
    for key, mods in sorted(group(runtime_unused).items()):
        print(f"\n[{key}] ({len(mods)} files)")
        for mod in mods:
            print(f"  {mod}")

    print("\n=== ONLY USED BY SCRIPTS/TESTS (not runtime) ===")
    print(f"Count: {len(repo_only)}")
    for key, mods in sorted(group(sorted(repo_only)).items()):
        print(f"\n[{key}] ({len(mods)} files)")
        for mod in mods:
            print(f"  {mod}")

    print("\n=== NEVER IMPORTED ANYWHERE (orphans) ===")
    print(f"Count: {len(orphans)}")
    for mod in orphans:
        print(f"  {mod}")

    folders: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "runtime": 0})
    for mod in all_mods:
        parts = mod.split(".")
        if len(parts) >= 2:
            folder = parts[1]
            folders[folder]["total"] += 1
            if mod in runtime:
                folders[folder]["runtime"] += 1

    print("\n=== TOP-LEVEL app/ FOLDERS (runtime / total) ===")
    for folder, counts in sorted(
        folders.items(),
        key=lambda item: item[1]["total"] - item[1]["runtime"],
        reverse=True,
    ):
        offline = counts["total"] - counts["runtime"]
        if offline:
            print(f"{folder}: {counts['runtime']}/{counts['total']} runtime ({offline} offline-only)")


if __name__ == "__main__":
    main()
