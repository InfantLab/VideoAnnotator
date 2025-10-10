#!/usr/bin/env python3
"""Conservative auto-fixer for common pydocstyle issues.

This script performs minimal, local-only edits:
- Normalize module-level docstrings that are short (convert to one-line form).
- Remove leading blank line inside multi-line docstrings.
- Ensure the first line of docstrings ends with a period (adds one if missing).
- For multi-line function docstrings, ensure there's a blank line between summary and description.

Run this from the repository root. It modifies files in-place. Review changes before committing.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = ["src", "scripts", "tests", "examples"]


def normalize_docstring_block(block: str) -> str:
    """Normalize a triple-quoted docstring block (including quotes).

    block includes the start and end triple quotes.
    Returns the normalized block (may be unchanged).
    """
    # detect the quoting style
    quote = block[:3]
    inner = block[3:-3]
    lines = inner.splitlines()
    # strip uniform leading/trailing blank lines
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()

    if not lines:
        return block

    # first non-empty line is the summary
    summary = lines[0].strip()
    # ensure summary ends with punctuation
    if summary and summary[-1] not in ".!?":
        summary = summary + "."

    if len(lines) == 1:
        # convert to single-line docstring
        return f"{quote}{summary}{quote}"

    # multi-line: ensure no leading blank line and one blank line between summary and rest
    rest = [ln.rstrip() for ln in lines[1:]]
    # remove leading blank lines in rest
    while rest and rest[0].strip() == "":
        rest.pop(0)

    # ensure there's exactly one blank line between summary and rest
    new_lines = [summary, ""] + rest
    inner2 = "\n".join(new_lines)
    return quote + inner2 + quote


def process_file(path: Path) -> bool:
    s = path.read_text(encoding="utf-8")
    orig = s

    # Fix module-level docstring at file start
    mod_doc_re = re.compile(r"\A(\s*)(['\"]{3})(.*?)(\2)", re.DOTALL)
    m = mod_doc_re.match(s)
    if m:
        block = s[m.start(2) : m.end(4)]
        new_block = normalize_docstring_block(block)
        if new_block != block:
            s = s[: m.start(2)] + new_block + s[m.end(4) :]

    # Fix simple function/method docstrings: look for def ...:\n    \s+"""
    func_doc_re = re.compile(
        r"(^\s*def\s+[\w_]+\s*\(.*?\):\n)(\s*)(['\"]{3})(.*?)(\3)",
        re.DOTALL | re.MULTILINE,
    )

    def _repl(m):
        pre = m.group(1)
        indent = m.group(2)
        block = m.group(3) + m.group(4) + m.group(5)
        new_block = normalize_docstring_block(block)
        return pre + indent + new_block

    s = func_doc_re.sub(_repl, s)

    if s != orig:
        path.write_text(s, encoding="utf-8")
        return True
    return False


def main():
    changed = []
    for d in TARGET_DIRS:
        p = PROJECT_ROOT / d
        if not p.exists():
            continue
        for file in p.rglob("*.py"):
            try:
                if process_file(file):
                    changed.append(str(file.relative_to(PROJECT_ROOT)))
            except Exception as e:
                print(f"[ERROR] Failed to process {file}: {e}")

    if changed:
        print("Modified files:")
        for c in changed:
            print(" -", c)
        print("Review changes and run pydocstyle again.")
        sys.exit(0)
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
