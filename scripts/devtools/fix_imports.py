#!/usr/bin/env python3
"""Fix absolute imports to relative imports after src/ → src/videoannotator/ migration.

This is a one-off maintenance script.
"""

from __future__ import annotations

import re
from pathlib import Path

MODULES = [
    "api",
    "auth",
    "batch",
    "database",
    "exporters",
    "pipelines",
    "registry",
    "schemas",
    "storage",
    "utils",
    "validation",
    "visualization",
    "worker",
    "config",
    "main",
    "cli",
    "version",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fix_file(file_path: Path) -> bool:
    """Fix imports in a single file. Returns True if changes were made."""
    content = file_path.read_text()
    original = content

    for module in MODULES:
        pattern = rf"^from {module}(\.[a-zA-Z0-9_.]+)? import "
        replacement = rf"from .{module}\1 import "
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content != original:
        file_path.write_text(content)
        return True
    return False


def main() -> None:
    src_dir = _repo_root() / "src" / "videoannotator"
    changed_files: list[Path] = []

    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            changed_files.append(py_file)
            rel = py_file.relative_to(_repo_root())
            print(f"Fixed: {rel}")

    print(f"\nTotal files changed: {len(changed_files)}")


if __name__ == "__main__":
    main()
