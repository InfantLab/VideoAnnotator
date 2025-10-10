"""Test that generated docs/pipelines_spec.md covers all registry pipelines.

This is a lightweight guard ensuring contributors regenerate the spec after
changing registry metadata."""

from pathlib import Path

from src.registry.pipeline_registry import get_registry

SPEC_PATH = Path("docs/pipelines_spec.md")


def test_all_registry_pipelines_present():
    registry = get_registry()
    registry.load(force=True)
    text = SPEC_PATH.read_text(encoding="utf-8")
    missing = []
    for meta in registry.list():
        # Simple presence check on the name column boundary markers
        if f"| {meta.name} |" not in text:
            missing.append(meta.name)
    assert not missing, (
        f"Pipelines missing from spec doc: {missing}. Run generate_pipeline_specs.py"
    )
