#!/usr/bin/env python3
"""List imported top-level modules within the repository.

Run:
  uv run python scripts/devtools/findimports.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_imports_in_file(file_path: str) -> list[str]:
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", content, re.MULTILINE)


def find_imports_in_directory(directory: str) -> set[str]:
    imports: set[str] = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports.update(find_imports_in_file(file_path))
    return imports


def main() -> None:
    project_directory = str(_repo_root())
    imports = find_imports_in_directory(project_directory)
    print("Identified packages:")
    for imp in sorted(imports):
        print(imp)


if __name__ == "__main__":
    main()
