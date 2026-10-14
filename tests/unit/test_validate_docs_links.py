"""scripts/validate_docs_links.py finds dead relative links and nothing else."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "validate_docs_links",
    Path(__file__).resolve().parents[2] / "scripts" / "validate_docs_links.py",
)
assert _spec and _spec.loader
links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(links)


def test_reports_only_the_dead_relative_links(tmp_path):
    (tmp_path / "docs" / "usage").mkdir(parents=True)
    (tmp_path / "docs" / "archive").mkdir()
    (tmp_path / "docs" / "usage" / "real file.md").write_text("# x")
    (tmp_path / "docs" / "img.png").write_bytes(b"")
    (tmp_path / "docs" / "README.md").write_text(
        "[ok](usage/real%20file.md#section) ![img](img.png) [web](https://x.org/nope)\n"
        "[anchor](#here) [mail](mailto:a@b)\n"
        "[gone](usage/missing.md)\n"
        '<img src="nope.png">\n'
        "`[code](not/checked.md)`\n"
        "```\n[fenced](not/checked.md)\n```\n"
        "[ref]: ../nowhere.md\n"
    )
    (tmp_path / "docs" / "archive" / "old.md").write_text("[dead](dead.md)")
    (tmp_path / "README.md").write_text("[docs](docs/README.md) [root](/docs/img.png)")

    assert links.broken_links(tmp_path) == [
        "docs/README.md:3: usage/missing.md",
        "docs/README.md:4: nope.png",
        "docs/README.md:9: ../nowhere.md",
    ]


def test_the_repository_docs_have_no_dead_links():
    root = Path(__file__).resolve().parents[2]
    assert links.broken_links(root) == []
