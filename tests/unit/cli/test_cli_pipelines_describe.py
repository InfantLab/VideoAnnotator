"""`videoannotator pipelines describe` (spec 026, contracts/cli.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from videoannotator.cli import app
from videoannotator.registry import pipeline_registry
from videoannotator.registry.pipeline_registry import PipelineRegistry

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "registry"


@pytest.fixture
def use_fixture(monkeypatch):
    def use(name: str):
        registry = PipelineRegistry(metadata_dir=FIXTURES / name)
        registry.load(force=True)
        monkeypatch.setattr(pipeline_registry, "get_registry", lambda: registry)

    return use


def test_builtin_text_starts_with_title_and_summary():
    result = runner.invoke(app, ["pipelines", "describe", "face_analysis"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "Faces: expression, age, gender  (face_analysis)"
    assert lines[1].startswith("Finds faces")
    assert "About" in lines


def test_builtin_json_has_guide_fields():
    result = runner.invoke(app, ["pipelines", "describe", "face_analysis", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["title"] == "Faces: expression, age, gender"
    assert {"summary", "homepage", "benchmarks", "quality_note", "citations"} <= set(
        data
    )
    assert data["citations"][0]["bibtex"].startswith("@")
    assert "group" in data["config_schema"]["detect_emotions"]


def test_unknown_pipeline_exits_with_a_hint():
    result = runner.invoke(app, ["pipelines", "describe", "no_such_pipeline"])
    assert result.exit_code == 1
    assert "videoannotator pipelines list" in result.output


def test_metadata_without_guide_falls_back(use_fixture):
    use_fixture("no_guide")
    result = runner.invoke(app, ["pipelines", "describe", "no_guide"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "No Guide Pipeline  (no_guide)"
    assert lines[1].startswith("Metadata written before spec 026")
    assert "Technical (1)" in lines
    assert "What to measure" not in lines
    assert "  Benchmark  not stated" in lines


def test_groups_units_and_conditions(use_fixture):
    use_fixture("full_guide")
    out = runner.invoke(app, ["pipelines", "describe", "full_guide"]).output
    measure = out.index("What to measure")
    technical = out.index("Technical (4)")
    most = out.index("Most things per frame")
    assert measure < most < technical
    assert "10 things (min 1, max 100)" in out
    assert "only when backend is slow" in out


def test_about_section(use_fixture):
    use_fixture("full_guide")
    out = runner.invoke(app, ["pipelines", "describe", "full_guide"]).output
    assert "  Homepage   https://example.org/method-x" in out
    assert "  Quality    Depends on the model you choose." in out
    assert "  Benchmark  Things benchmark (hard), AP 91.4% (large backbone)" in out
    assert "             https://example.org/paper" in out


def test_empty_benchmarks_says_none_published(tmp_path, monkeypatch):
    text = (FIXTURES / "no_guide" / "no_guide.yaml").read_text()
    (tmp_path / "no_guide.yaml").write_text(text + "benchmarks: []\n")
    registry = PipelineRegistry(metadata_dir=tmp_path)
    registry.load(force=True)
    monkeypatch.setattr(pipeline_registry, "get_registry", lambda: registry)
    out = runner.invoke(app, ["pipelines", "describe", "no_guide"]).output
    assert "  Benchmark  No published benchmark" in out
