"""Guard: nothing in the shipped metadata directory points at test code
(specs/011-pipeline-readiness FR-004). A stray copy of the test fixture
stub_pipeline.yaml there once made "Stub Forward-Compatibility Pipeline" the
only pipeline a core-only user saw."""

from pathlib import Path

import yaml

import videoannotator.registry as registry_pkg

METADATA_DIR = Path(registry_pkg.__file__).parent / "metadata"


def test_no_shipped_pipeline_is_a_test_fixture():
    offenders = []
    for path in sorted(METADATA_DIR.glob("*.y*ml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        module_path = str(raw.get("module_path", ""))
        if module_path.startswith("tests.") or raw.get("pipeline_family") == "stub":
            offenders.append(path.name)
    assert offenders == []
