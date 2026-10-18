"""How to cite VideoAnnotator itself, for the methods paragraph (spec 025).

Kept here rather than read from CITATION.cff, which an installed package
doesn't ship; tests/unit/registry/test_citations.py keeps the two in step.
"""

from __future__ import annotations

KEY = "addyman2026videoannotator"
TITLE = (
    "VideoAnnotator: an extensible, reproducible toolkit for automated video "
    "annotation in behavioral research"
)
AUTHORS = (
    ("Addyman", "Caspar"),
    ("Ishaya", "Jeremiah"),
    ("Uwerikowe", "Irene"),
    ("Stamate", "Daniel"),
    ("Lachman", "Jamie"),
    ("Tomlinson", "Mark"),
)
REPOSITORY = "https://github.com/InfantLab/VideoAnnotator"


def videoannotator_bibtex(version: str | None) -> str:
    """VideoAnnotator's BibTeX entry, for the version that made the results."""
    authors = " and ".join(f"{family}, {given}" for family, given in AUTHORS)
    lines = [
        f"@software{{{KEY},",
        f"  author = {{{authors}}},",
        f"  title = {{{{VideoAnnotator}}: {TITLE.split(': ', 1)[1]}}},",
    ]
    if version:
        lines.append(f"  version = {{{version}}},")
    lines += [
        "  year = {2026},",
        f"  url = {{{REPOSITORY}}},",
        "  license = {MIT}",
        "}",
    ]
    return "\n".join(lines)
