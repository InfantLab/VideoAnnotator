"""Every shipped pipeline says what to cite (spec 025)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from videoannotator.exporters import citation
from videoannotator.registry.pipeline_registry import PipelineRegistry

REPO = Path(__file__).resolve().parents[3]
ENTRY = re.compile(r"^@(\w+)\{([^,\s]+),", re.MULTILINE)


def _pipelines():
    registry = PipelineRegistry()
    registry.load(force=True)
    return registry.list(include_deprecated=True)


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth < 0:
            return False
    return depth == 0


@pytest.mark.parametrize("meta", _pipelines(), ids=lambda m: m.name)
def test_each_pipeline_has_well_formed_citations(meta):
    assert meta.citations, f"{meta.name} has no citations"
    for c in meta.citations:
        entries = ENTRY.findall(c.bibtex)
        assert len(entries) == 1, c.key
        assert entries[0][1] == c.key
        assert _balanced(c.bibtex), c.key


def test_a_key_always_means_the_same_entry():
    seen: dict[str, str] = {}
    for meta in _pipelines():
        for c in meta.citations:
            assert seen.setdefault(c.key, c.bibtex) == c.bibtex, c.key


def test_videoannotator_entry_matches_citation_cff():
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["title"] == citation.TITLE
    assert cff["repository-code"] == citation.REPOSITORY
    assert list(citation.AUTHORS) == [
        (a["family-names"], a["given-names"]) for a in cff["authors"]
    ]
    entry = citation.videoannotator_bibtex("1.6.0")
    assert ENTRY.findall(entry) == [("software", citation.KEY)]
    assert "version = {1.6.0}" in entry and _balanced(entry)
