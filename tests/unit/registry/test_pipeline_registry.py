"""Unit tests for extras-based metadata parsing and availability checking.

Covers: `requires_extras` parsing (data-model.md), `module_path`-required
validation (contracts/pipeline-metadata-schema.md), and the
extras-availability helper in `pipeline_loader.py` (research.md §3).
"""

import logging

from videoannotator.registry import pipeline_loader
from videoannotator.registry.pipeline_registry import PipelineRegistry

VALID_YAML = """\
name: {name}
display_name: Test Pipeline
description: A test pipeline.
{module_path_line}
{requires_extras_line}
outputs:
  - format: JSON
    types: [test]
config_schema:
  foo:
    type: string
    default: bar
version: 1
"""


def _write_metadata(
    tmp_path,
    filename,
    name="test_pipeline",
    module_path="videoannotator.pipelines.scene_detection:SceneDetectionPipeline",
    requires_extras=None,
):
    module_path_line = f"module_path: {module_path}" if module_path else ""
    requires_extras_line = (
        f"requires_extras: [{', '.join(requires_extras)}]"
        if requires_extras is not None
        else ""
    )
    content = VALID_YAML.format(
        name=name,
        module_path_line=module_path_line,
        requires_extras_line=requires_extras_line,
    )
    (tmp_path / filename).write_text(content)


class TestRequiresExtrasParsing:
    def test_parses_requires_extras_list(self, tmp_path):
        _write_metadata(tmp_path, "test.yaml", requires_extras=["scene", "extra2"])
        reg = PipelineRegistry(metadata_dir=tmp_path)
        reg.load(force=True)
        meta = reg.get("test_pipeline")

        assert meta is not None
        assert meta.requires_extras == ["scene", "extra2"]
        assert meta.module_path == (
            "videoannotator.pipelines.scene_detection:SceneDetectionPipeline"
        )

    def test_requires_extras_absent_defaults_to_empty_list(self, tmp_path):
        _write_metadata(tmp_path, "test.yaml", requires_extras=None)
        reg = PipelineRegistry(metadata_dir=tmp_path)
        reg.load(force=True)
        meta = reg.get("test_pipeline")

        assert meta is not None
        assert meta.requires_extras == []

    def test_requires_extras_empty_list_is_valid(self, tmp_path):
        _write_metadata(tmp_path, "test.yaml", requires_extras=[])
        reg = PipelineRegistry(metadata_dir=tmp_path)
        reg.load(force=True)
        meta = reg.get("test_pipeline")

        assert meta is not None
        assert meta.requires_extras == []


class TestModulePathRequired:
    def test_missing_module_path_is_skipped_with_warning(self, tmp_path, caplog):
        _write_metadata(tmp_path, "bad.yaml", module_path=None)
        reg = PipelineRegistry(metadata_dir=tmp_path)

        with caplog.at_level(logging.WARNING):
            reg.load(force=True)

        assert reg.get("test_pipeline") is None
        assert any("module_path" in record.message for record in caplog.records)

    def test_present_module_path_loads_successfully(self, tmp_path):
        _write_metadata(tmp_path, "good.yaml")
        reg = PipelineRegistry(metadata_dir=tmp_path)
        reg.load(force=True)

        assert reg.get("test_pipeline") is not None


class TestExtrasAvailabilityHelper:
    def test_empty_requires_extras_is_vacuously_available(self):
        assert pipeline_loader.extras_available([]) is True

    def test_extras_available_false_when_a_package_missing(self, monkeypatch):
        monkeypatch.setattr(
            pipeline_loader,
            "_packages_for_extra",
            lambda extra: ("totally_fake_pkg_xyz",) if extra == "needs_pkg" else (),
        )
        monkeypatch.setattr(
            pipeline_loader,
            "_is_distribution_installed",
            lambda name: name != "totally_fake_pkg_xyz",
        )

        assert pipeline_loader.extras_available(["needs_pkg"]) is False
        assert pipeline_loader.extras_available(["other_group"]) is True

    def test_missing_extras_returns_only_unavailable_groups(self, monkeypatch):
        monkeypatch.setattr(
            pipeline_loader,
            "_packages_for_extra",
            lambda extra: ("missing_pkg",) if extra == "grp" else ("present_pkg",),
        )
        monkeypatch.setattr(
            pipeline_loader,
            "_is_distribution_installed",
            lambda name: name != "missing_pkg",
        )

        assert pipeline_loader.missing_extras(["grp", "other"]) == ["grp"]

    def test_install_hint_names_exact_pip_command(self):
        assert (
            pipeline_loader.install_hint(["face-laion"])
            == "pip install videoannotator[face-laion]"
        )
        assert (
            pipeline_loader.install_hint(["face-laion", "audio"])
            == "pip install videoannotator[face-laion,audio]"
        )
