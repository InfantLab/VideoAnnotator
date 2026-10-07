#!/usr/bin/env python3
"""Fail when a relative link in the docs points at a file that doesn't exist.

Checks README.md, CONTRIBUTING.md and docs/ (except docs/archive/, which keeps
historical documents as they were). Web links aren't fetched: they change for
reasons outside the repository and would make CI flaky.

Usage: python scripts/validate_docs_links.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(
    r"^[ \t]*\[[^\]]+\]:[ \t]*<?(\S+?)>?(?:[ \t]+\".*\")?[ \t]*$", re.MULTILINE
)
HTML_SRC = re.compile(r"""<(?:img|a)\b[^>]*?\b(?:src|href)=["']([^"']+)["']""")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.IGNORECASE)


def files_to_check(root: Path) -> list[Path]:
    files = [root / name for name in ("README.md", "CONTRIBUTING.md")]
    docs = root / "docs"
    files += [
        p for p in docs.rglob("*.md") if "archive" not in p.relative_to(docs).parts
    ]
    return sorted(p for p in files if p.is_file())


def strip_code(text: str) -> str:
    """Blank out fenced and inline code, keeping line numbers."""
    out, fenced = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            fenced = not fenced
            out.append("")
        else:
            out.append("" if fenced else INLINE_CODE.sub("", line))
    return "\n".join(out)


def targets(text: str) -> list[tuple[int, str]]:
    text = strip_code(text)
    found = []
    for pattern in (LINK, REFERENCE, HTML_SRC):
        for match in pattern.finditer(text):
            found.append((text.count("\n", 0, match.start(1)) + 1, match.group(1)))
    return sorted(found)


def broken_links(root: Path) -> list[str]:
    problems = []
    for path in files_to_check(root):
        for line, target in targets(path.read_text(encoding="utf-8")):
            if EXTERNAL.match(target):
                continue
            local = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not local:
                continue
            resolved = (
                (root / local.lstrip("/"))
                if local.startswith("/")
                else path.parent / local
            )
            if not resolved.exists():
                problems.append(f"{path.relative_to(root)}:{line}: {target}")
    return problems


def main() -> int:
    root = Path(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    )
    problems = broken_links(root)
    for problem in problems:
        print(problem)
    checked = len(files_to_check(root))
    if problems:
        print(f"\n{len(problems)} broken link(s) in {checked} files.")
        return 1
    print(f"All relative links resolve ({checked} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
