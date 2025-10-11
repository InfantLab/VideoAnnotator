"""AST-based docstring normalizer using libcst.

This script performs conservative, structural docstring normalizations:
- Ensure multi-line docstrings have closing triple-quotes on their own line.
- Ensure the first summary line ends with punctuation.
- Normalize indentation for Args/Returns sections (align to code indent + 4 spaces).

Run from repository root. It prints modified files and applies changes in-place.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import libcst as cst
from libcst import MetadataWrapper
from libcst.metadata import PositionProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = ["src", "tests"]


def normalize_docstring_text(text: str) -> str:
    # Strip surrounding quotes if present (we only get the content here normally)
    lines = text.splitlines()
    # Remove leading/trailing blank lines
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()

    if not lines:
        return text

    # Ensure first line ends with punctuation
    first = lines[0].rstrip()
    if first and first[-1] not in ".!?":
        first = first + "."
    lines[0] = first

    # Ensure multi-line docstrings end with a blank line before closing quotes
    if len(lines) > 1 and lines[-1].strip() != "":
        lines.append("")

    # Normalize Arg/Return section indentation heuristically
    out = []
    in_section = False
    for ln in lines:
        if re.match(r"^\s*(Args|Arguments|Returns|Raises|Yields|Examples):", ln):
            # normalize section header to left (no extra indent here) — caller will re-indent
            out.append(ln.strip())
            in_section = True
            continue

        if in_section:
            # indent section body by 4 spaces (we'll rely on docformatter later for final reflow)
            if ln.strip() == "":
                out.append("")
            else:
                out.append("    " + ln.lstrip())
            # keep in_section until we hit a blank line followed by non-indented text
            if ln.strip() == "":
                in_section = False
            continue

        out.append(ln)

    return "\n".join(out)


class DocstringTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.BaseStatement:
        # Look for Expr(Str) which is a module-level docstring or simple docstring statements
        if len(original.body) == 1 and isinstance(original.body[0], cst.Expr):
            expr = original.body[0].value
            if isinstance(expr, cst.SimpleString):
                val = expr.evaluated_value  # this gives the content without quotes
                new_val = normalize_docstring_text(val)
                if new_val != val:
                    # reconstruct string with triple quotes and keep original quote style (use triple double)
                    new_src = '"""' + new_val + '"""'
                    return updated.with_changes(
                        body=[cst.Expr(cst.SimpleString(value=new_src))]
                    )
        return updated

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        # Handle function-level docstrings in the body first statement
        body = original.body.body
        if body and isinstance(body[0], cst.SimpleStatementLine):
            stmt = body[0]
            if stmt.body and isinstance(stmt.body[0], cst.Expr):
                expr = stmt.body[0].value
                if isinstance(expr, cst.SimpleString):
                    val = expr.evaluated_value
                    new_val = normalize_docstring_text(val)
                    if new_val != val:
                        new_src = '"""' + new_val + '"""'
                        new_stmt = stmt.with_changes(
                            body=[cst.Expr(cst.SimpleString(value=new_src))]
                        )
                        new_body = [new_stmt] + list(body[1:])
                        return updated.with_changes(
                            body=updated.body.with_changes(body=new_body)
                        )
        return updated

    def leave_ClassDef(
        self, original: cst.ClassDef, updated: cst.ClassDef
    ) -> cst.ClassDef:
        # Handle class-level docstrings similarly
        body = original.body.body
        if body and isinstance(body[0], cst.SimpleStatementLine):
            stmt = body[0]
            if stmt.body and isinstance(stmt.body[0], cst.Expr):
                expr = stmt.body[0].value
                if isinstance(expr, cst.SimpleString):
                    val = expr.evaluated_value
                    new_val = normalize_docstring_text(val)
                    if new_val != val:
                        new_src = '"""' + new_val + '"""'
                        new_stmt = stmt.with_changes(
                            body=[cst.Expr(cst.SimpleString(value=new_src))]
                        )
                        new_body = [new_stmt] + list(body[1:])
                        return updated.with_changes(
                            body=updated.body.with_changes(body=new_body)
                        )
        return updated


def process_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    wrapper = MetadataWrapper(cst.parse_module(src))
    mod = wrapper.module
    transformer = DocstringTransformer()
    new_mod = mod.visit(transformer)
    new_src = new_mod.code
    if new_src != src:
        path.write_text(new_src, encoding="utf-8")
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
                print(f"[ERROR] {file}: {e}")

    if changed:
        print("Modified files:")
        for c in changed:
            print(" -", c)
        sys.exit(0)
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
