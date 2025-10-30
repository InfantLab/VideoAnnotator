#!/usr/bin/env python3
"""Fix absolute imports to relative imports after src/ → src/videoannotator/ migration."""

import re
from pathlib import Path

# Modules that were moved into videoannotator/
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


def fix_file(file_path: Path) -> bool:
    """Fix imports in a single file. Returns True if changes were made."""
    content = file_path.read_text()
    original = content

    # Pattern: from MODULE import ... or from MODULE.submodule import ...
    for module in MODULES:
        # Match: from module import X or from module.submodule import X
        pattern = rf"^from {module}(\.[a-zA-Z0-9_.]+)? import "
        replacement = rf"from .{module}\1 import "
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content != original:
        file_path.write_text(content)
        return True
    return False


def main():
    src_dir = Path("src/videoannotator")
    changed_files = []

    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            changed_files.append(py_file)
            print(f"Fixed: {py_file.relative_to(src_dir.parent)}")

    print(f"\nTotal files changed: {len(changed_files)}")


if __name__ == "__main__":
    main()
