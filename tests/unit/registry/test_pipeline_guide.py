"""The pipeline guide (spec 026): parsing, and the built-ins' completeness."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml

from videoannotator.registry.pipeline_registry import PipelineRegistry

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "registry"


def _load(name: str):
    registry = PipelineRegistry(metadata_dir=FIXTURES / name)
    registry.load(force=True)
    meta = registry.get(name)
    assert meta is not None
    return meta


def test_metadata_without_guide_keys_loads_with_fallbacks():
    meta = _load("no_guide")
    assert meta.title is None
    assert meta.guide_title == "No Guide Pipeline"
    assert meta.guide_summary == meta.description
    assert meta.homepage is None
    assert meta.benchmarks is None
    assert meta.quality_note is None
    field = meta.config_schema["threshold"]
    assert field.group == "technical"
    assert field.label is None
    assert field.applies_when is None


def test_full_guide_is_parsed():
    meta = _load("full_guide")
    assert meta.guide_title == "Things: what they are"
    assert meta.summary.startswith("Finds things")
    assert meta.homepage == "https://example.org/method-x"
    assert meta.quality_note == "Depends on the model you choose."
    most = meta.config_schema["most_things"]
    assert (most.group, most.label, most.unit, most.min, most.max) == (
        "behavioural",
        "Most things per frame",
        "things",
        1,
        100,
    )
    assert meta.config_schema["slow_only"].applies_when == {"backend": ["slow"]}


def test_benchmark_figure_is_kept_as_published_and_bad_entries_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger="videoannotator.pipelines"):
        meta = _load("full_guide")
    assert meta.benchmarks is not None
    assert [b.name for b in meta.benchmarks] == ["Things benchmark (hard)"]
    assert meta.benchmarks[0].figure == "91.4%"
    assert meta.benchmarks[0].note == "large backbone"
    assert "invalid benchmarks entry" in caplog.text


def test_unknown_group_is_technical_and_dangling_condition_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger="videoannotator.pipelines"):
        meta = _load("full_guide")
    assert meta.config_schema["odd_group"].group == "technical"
    assert meta.config_schema["dangling"].applies_when is None
    assert "unknown group" in caplog.text
    assert "applies_when names an unknown setting" in caplog.text


def test_existing_fields_unchanged_by_guide_keys():
    meta = _load("full_guide")
    backend = meta.config_schema["backend"]
    assert (backend.type, backend.default, backend.enum) == (
        "string",
        "fast",
        ["fast", "slow"],
    )
    assert meta.display_name == "Full Guide Pipeline (Method X)"


@pytest.mark.parametrize("raw", [[], "not a list"])
def test_empty_benchmarks_means_none_published(tmp_path, raw):
    text = (FIXTURES / "no_guide" / "no_guide.yaml").read_text()
    value = "[]" if raw == [] else '"not a list"'
    (tmp_path / "no_guide.yaml").write_text(text + f"benchmarks: {value}\n")
    registry = PipelineRegistry(metadata_dir=tmp_path)
    registry.load(force=True)
    meta = registry.get("no_guide")
    assert meta.benchmarks == ([] if raw == [] else None)


# --- The built-in pipelines' guides (spec 026 FR-004, FR-011, SC-001, SC-002) ---

REPO = Path(__file__).resolve().parents[3]
METADATA = REPO / "src" / "videoannotator" / "registry" / "metadata"
BUILTINS = sorted(METADATA.glob("*.yaml"))


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _package_text(raw: dict) -> str:
    module = raw["module_path"].split(":")[0]
    package = REPO / "src" / Path(*module.split(".")[:-1])
    return "\n".join(p.read_text() for p in package.rglob("*.py"))


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_every_setting_is_grouped_and_labelled(path):
    raw = _raw(path)
    for key, field in raw["config_schema"].items():
        assert field.get("group") in ("behavioural", "technical"), key
        assert field.get("label"), key


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_at_most_five_behavioural_settings(path):
    raw = _raw(path)
    behavioural = [
        k for k, f in raw["config_schema"].items() if f.get("group") == "behavioural"
    ]
    assert len(behavioural) <= 5, behavioural


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_conditions_name_real_settings(path):
    schema = _raw(path)["config_schema"]
    for key, field in schema.items():
        for other, values in (field.get("applies_when") or {}).items():
            assert other in schema, (key, other)
            enum = schema[other].get("enum")
            if enum:
                assert set(values) <= set(enum), (key, other, values)


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_every_setting_is_read_by_its_pipeline(path):
    raw = _raw(path)
    text = _package_text(raw)
    unread = [
        k for k in raw["config_schema"] if not re.search(rf"\b{re.escape(k)}\b", text)
    ]
    assert not unread, f"{path.stem} declares settings its code never reads: {unread}"


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_title_and_short_summary(path):
    raw = _raw(path)
    assert raw.get("title"), path.stem
    sentences = [
        s for s in re.split(r"(?<=[.!?])\s+", raw.get("summary", "").strip()) if s
    ]
    assert 1 <= len(sentences) <= 2, (path.stem, sentences)


@pytest.mark.parametrize("path", BUILTINS, ids=lambda p: p.stem)
def test_homepage_and_benchmarks_stated(path):
    raw = _raw(path)
    assert str(raw.get("homepage", "")).startswith("https://"), path.stem
    assert "benchmarks" in raw, (
        f"{path.stem}: say `benchmarks: []` if none is published"
    )
    for b in raw["benchmarks"]:
        assert all(b.get(f) for f in ("name", "measure", "figure", "source")), b
        assert isinstance(b["figure"], str), b
        assert str(b["source"]).startswith("https://"), b
    if not raw["benchmarks"]:
        assert raw.get("quality_note"), (
            f"{path.stem}: an empty benchmark list needs a quality note saying why"
        )


def test_api_cli_agree():
    """SC-004: the API and `pipelines describe --json` give the same guide."""
    import json

    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from videoannotator.api.main import create_app
    from videoannotator.cli import app as cli_app

    client = TestClient(create_app())
    runner = CliRunner()
    keys = ("title", "summary", "homepage", "benchmarks", "quality_note", "citations")
    for path in BUILTINS:
        name = _raw(path)["name"]
        api = client.get(f"/api/v1/pipelines/{name}").json()
        cli = json.loads(
            runner.invoke(cli_app, ["pipelines", "describe", name, "--json"]).output
        )
        for key in keys:
            assert api[key] == cli[key], (name, key)
        assert api["config_schema"] == cli["config_schema"], name
