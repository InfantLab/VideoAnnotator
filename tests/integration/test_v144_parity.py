"""Integration test: v1.4.4 -> v1.5.0 output parity under an `[all]` install
(User Story 2, FR-009, quickstart.md §3).

Deterministic pipelines (e.g. scene_detection's shot-boundary detection,
person_tracking's detection/tracking given a fixed seed) must produce
byte-identical output to the v1.4.4 baseline; non-deterministic ones
(anything involving sampling or backend-dependent floating point, e.g.
Whisper decoding or pyannote clustering) get a documented relative
tolerance instead of exact equality.

Golden fixtures would live in tests/fixtures/v144_golden/<pipeline>.json,
captured once from the `v1.4.4` git tag in a fully-provisioned `[all]`
environment (real model downloads + inference — not something to do as
part of routine `pytest tests/`):

    git worktree add /tmp/va-v144 v1.4.4
    cd /tmp/va-v144 && uv sync --all-extras
    # run each pipeline against a fixed sample video/config, serialize its
    # output to tests/fixtures/v144_golden/<pipeline>.json

No such fixtures exist yet in this repo, and no runner that produces
comparable current-codebase output exists yet either — both are follow-up
work. The comparison tests below are structurally complete (they'll run
for real the moment fixtures + a runner are added) and skip themselves
with a specific reason until then, rather than asserting something
nothing was actually verified against. The tolerance-comparison helper
itself is exercised directly by a real, always-on unit test.
"""

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "v144_golden"

# FR-009: pipelines whose output is deterministic given the same input/config
# get byte-identical comparison; everything else gets a documented tolerance.
DETERMINISTIC_PIPELINES = {"scene_detection", "person_tracking"}
NON_DETERMINISTIC_TOLERANCE = {
    # pipeline_name: (relative float tolerance, rationale)
    "audio_processing": (1e-3, "Whisper decoding can vary slightly by backend/BLAS"),
    "speech_recognition": (1e-3, "same as audio_processing"),
    "speaker_diarization": (1e-2, "pyannote clustering has run-to-run jitter"),
}


def _run_pipeline_for_parity(pipeline_name: str):
    """Run `pipeline_name` against the shared parity sample video/config and
    return its JSON-serializable output, for comparison against a v1.4.4
    golden fixture.

    Not implemented: no golden fixtures exist yet to compare against (see
    module docstring), so this is never called in practice today. Building
    it is a prerequisite for adding real fixtures, not optional scaffolding.
    """
    raise NotImplementedError(
        "No golden-fixture capture/comparison tooling exists yet. To add "
        "real v1.4.4 parity coverage: check out the `v1.4.4` git tag into a "
        "fully-provisioned `[all]` environment, run "
        f"'{pipeline_name}' against a fixed sample video, serialize its "
        f"output to tests/fixtures/v144_golden/{pipeline_name}.json, and "
        "implement this function to produce the same shape from the "
        "current codebase."
    )


def _assert_within_tolerance(current, golden, tolerance: float):
    """Recursively compare `current` vs `golden`: `tolerance` relative
    error on float leaves, exact equality everywhere else."""
    if isinstance(golden, dict):
        assert isinstance(current, dict) and current.keys() == golden.keys()
        for key in golden:
            _assert_within_tolerance(current[key], golden[key], tolerance)
    elif isinstance(golden, list):
        assert isinstance(current, list) and len(current) == len(golden)
        for c, g in zip(current, golden, strict=True):
            _assert_within_tolerance(c, g, tolerance)
    elif isinstance(golden, float):
        assert current == pytest.approx(golden, rel=tolerance)
    else:
        assert current == golden


class TestToleranceComparisonHelper:
    """Real, always-on coverage of the comparison logic itself, independent
    of whether golden fixtures exist yet."""

    def test_identical_structures_pass(self):
        data = {"a": 1, "b": [1.0, 2.0], "c": "text"}
        _assert_within_tolerance(data, data, tolerance=1e-6)

    def test_float_within_tolerance_passes(self):
        golden = {"score": 0.700000}
        current = {"score": 0.700049}  # ~0.007% relative difference
        _assert_within_tolerance(current, golden, tolerance=1e-3)

    def test_float_outside_tolerance_fails(self):
        golden = {"score": 0.7}
        current = {"score": 0.9}
        with pytest.raises(AssertionError):
            _assert_within_tolerance(current, golden, tolerance=1e-3)

    def test_non_float_mismatch_fails_exactly(self):
        golden = {"label": "cat"}
        current = {"label": "dog"}
        with pytest.raises(AssertionError):
            _assert_within_tolerance(current, golden, tolerance=1e-3)


class TestV144Parity:
    def test_golden_fixtures_documented_but_not_yet_captured(self):
        fixtures = sorted(GOLDEN_DIR.glob("*.json")) if GOLDEN_DIR.exists() else []
        if not fixtures:
            pytest.skip(
                f"No v1.4.4 golden fixtures in {GOLDEN_DIR} yet — see this "
                "file's module docstring for how to capture them from the "
                "v1.4.4 git tag. Nothing to compare against until then."
            )

    @pytest.mark.parametrize("pipeline_name", sorted(DETERMINISTIC_PIPELINES))
    def test_deterministic_pipeline_matches_v144_byte_identical(self, pipeline_name):
        fixture_path = GOLDEN_DIR / f"{pipeline_name}.json"
        if not fixture_path.exists():
            pytest.skip(
                f"No golden fixture for '{pipeline_name}' at {fixture_path} yet"
            )

        golden = json.loads(fixture_path.read_text())
        current = _run_pipeline_for_parity(pipeline_name)

        assert current == golden, (
            f"'{pipeline_name}' is a deterministic pipeline (FR-009) — its "
            "v1.5.0 output must be byte-identical to the v1.4.4 golden fixture."
        )

    @pytest.mark.parametrize("pipeline_name", sorted(NON_DETERMINISTIC_TOLERANCE))
    def test_non_deterministic_pipeline_matches_v144_within_tolerance(
        self, pipeline_name
    ):
        fixture_path = GOLDEN_DIR / f"{pipeline_name}.json"
        if not fixture_path.exists():
            pytest.skip(
                f"No golden fixture for '{pipeline_name}' at {fixture_path} yet"
            )

        tolerance, _rationale = NON_DETERMINISTIC_TOLERANCE[pipeline_name]
        golden = json.loads(fixture_path.read_text())
        current = _run_pipeline_for_parity(pipeline_name)

        _assert_within_tolerance(current, golden, tolerance)
