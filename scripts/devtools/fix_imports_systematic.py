#!/usr/bin/env python3
"""Systematically fix imports from old structure to new videoannotator.* structure."""

from __future__ import annotations

import re
from pathlib import Path

MODULES_TO_FIX = [
    "api",
    "storage",
    "pipelines",
    "utils",
    "exporters",
    "schemas",
    "version",
    "logging_config",
    "config_env",
    "validation",
    "database",
    "batch",
    "worker",
    "auth",
]


IMPORT_FROM_PATTERN = re.compile(
    r"^from (" + "|".join(MODULES_TO_FIX) + r")(\.|\ import)", re.MULTILINE
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fix_imports_in_file(file_path: Path) -> bool:
    """Fix imports in a single file. Returns True if changes were made."""
    content = file_path.read_text()
    original_content = content

    def replace_import(match: re.Match[str]) -> str:
        module = match.group(1)
        rest = match.group(2)
        return f"from videoannotator.{module}{rest}"

    content = IMPORT_FROM_PATTERN.sub(replace_import, content)

    if content != original_content:
        file_path.write_text(content)
        print(f"[OK] Fixed: {file_path.relative_to(_repo_root())}")
        return True
    return False


def main() -> None:
    """Fix imports in test, script, and example files."""
    base = _repo_root()
    dirs_to_scan = [base / "tests", base / "scripts", base / "examples"]

    fixed_count = 0
    for directory in dirs_to_scan:
        if not directory.exists():
            continue
        for py_file in directory.rglob("*.py"):
            if fix_imports_in_file(py_file):
                fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == "__main__":
    main()
