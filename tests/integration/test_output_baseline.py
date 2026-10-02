"""Integration test: pipeline outputs on the demo clip match the committed baseline.

The baseline is the viewer contract fixtures (tests/fixtures/viewer_contract/), so one
set of real outputs guards both sides: the viewer reading them, and the pipelines still
producing them. This test runs the demo clip through a real server, the same path a
user's job takes, and compares each output file with its fixture.

Re-baselined 2026-10-02 (v1.6.0 Phase 1). v1.5.0 and 1.6-dev (Python 3.13, torch 2.11,
pyannote.audio 4) produce the same outputs up to run-to-run GPU noise; see CHANGELOG.
The v1.4.4 parity fixtures this file used to wait for were never captured; the v1.5.0
comparison replaced them.

Needs every pipeline's extras and model weights, and takes about a minute on a GPU,
so it is marked real_models (deselected in CI). When an output changes on purpose,
regenerate the fixtures as tests/fixtures/viewer_contract/README.md describes.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "tests" / "fixtures" / "viewer_contract"
DEMO_VIDEO = (
    REPO_ROOT / "viewer" / "demo-assets" / "2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4"
)
# Fixture names and contents replace the participant code (viewer_contract/README.md).
RENAMES = [("2UWdXP.joke1.rep3.take1.Peekaboo_h265", "demo_clip"), ("2UWdXP", "demo")]

# face_analysis is left out: DeepFace detects no faces in the demo clip, so it writes
# no file to compare.
PIPELINES = [
    "face_openface3_embedding",
    "person_tracking",
    "scene_detection",
    "speaker_diarization",
    "speech_recognition",
]

EXACT = [
    "scene_detection.json",
    "person_tracks.json",
    "speaker_diarization.rttm",
    "speech_recognition.vtt",
]
# (rel, abs) per file. Measured 2026-10-02 on an RTX 4060, two runs of the same code:
# person scores differ by up to 0.07%, OpenFace 3 action-unit intensities by up to
# 1.6% (abs 0.004). Bounds are about 10x that; a CPU run may need more.
TOLERANCE = {
    "person_tracking.json": (1e-2, 1e-3),
    "openface3_analysis.json": (5e-2, 1e-2),
    "openface3_detailed.json": (5e-2, 1e-2),
}


def _assert_within_tolerance(current, golden, rel: float, abs_: float = 0.0, path=""):
    """Recursively compare `current` vs `golden`: approximate on numbers, exact
    everywhere else."""
    if isinstance(golden, dict):
        assert isinstance(current, dict), path
        assert current.keys() == golden.keys(), path
        for key in golden:
            _assert_within_tolerance(
                current[key], golden[key], rel, abs_, f"{path}/{key}"
            )
    elif isinstance(golden, list):
        assert isinstance(current, list), path
        assert len(current) == len(golden), path
        for i, (c, g) in enumerate(zip(current, golden, strict=True)):
            _assert_within_tolerance(c, g, rel, abs_, f"{path}[{i}]")
    elif isinstance(golden, int | float) and not isinstance(golden, bool):
        assert current == pytest.approx(golden, rel=rel, abs=abs_), path
    else:
        assert current == golden, path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def current_outputs(tmp_path_factory) -> dict[str, str]:
    """Run the demo clip through a real server; return {fixture suffix: content}."""
    work = tmp_path_factory.mktemp("output_baseline")
    port = _free_port()
    base = f"http://127.0.0.1:{port}/api/v1"
    env = {
        **os.environ,
        "VIDEOANNOTATOR_DB_PATH": str(work / "va.db"),
        "STORAGE_ROOT": str(work / "storage"),
        "STORAGE_BASE_DIR": str(work / "batch"),
        "AUTH_REQUIRED": "false",
        # tests/conftest.py turns this off for the suite; the server must process jobs.
        "VIDEOANNOTATOR_BACKGROUND_PROCESSING": "true",
    }
    cli = Path(sys.executable).parent / "videoannotator"
    with (work / "server.log").open("w") as log:
        server = subprocess.Popen(
            [str(cli), "server", "--port", str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        for _ in range(120):
            try:
                requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except requests.ConnectionError:
                time.sleep(1)
        available = {
            p["name"]: p.get("available", True)
            for p in requests.get(f"{base}/pipelines/", timeout=30).json()["pipelines"]
        }
        missing = [p for p in PIPELINES if not available.get(p)]
        if missing:
            pytest.skip(f"Pipelines not installed: {', '.join(missing)}")

        with DEMO_VIDEO.open("rb") as video:
            job = requests.post(
                f"{base}/jobs/",
                files={"video": (DEMO_VIDEO.name, video, "video/mp4")},
                data={"selected_pipelines": ",".join(PIPELINES)},
                timeout=60,
            ).json()
        deadline = time.monotonic() + 15 * 60
        while time.monotonic() < deadline:
            status = requests.get(f"{base}/jobs/{job['id']}", timeout=30).json()
            if status["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)
        assert status["status"] == "completed", status.get("error_message")
    finally:
        server.terminate()
        server.wait(timeout=60)

    outputs = {}
    for path in (work / "storage" / job["id"]).iterdir():
        if path.suffix not in (".json", ".rttm", ".vtt"):
            continue
        name, text = path.name, path.read_text()
        for old, new in RENAMES:
            name, text = name.replace(old, new), text.replace(old, new)
        outputs[name.removeprefix("demo_clip_")] = text
    return outputs


@pytest.mark.parametrize("suffix", EXACT)
@pytest.mark.integration
@pytest.mark.real_models
@pytest.mark.slow
def test_output_matches_baseline_exactly(current_outputs, suffix):
    golden = (BASELINE_DIR / f"demo_clip_{suffix}").read_text()
    assert suffix in current_outputs, f"no {suffix} written"
    # The fixtures went through pre-commit's end-of-file fixer.
    assert current_outputs[suffix].rstrip("\n") == golden.rstrip("\n")


@pytest.mark.parametrize("suffix", sorted(TOLERANCE))
@pytest.mark.integration
@pytest.mark.real_models
@pytest.mark.slow
def test_output_matches_baseline_within_tolerance(current_outputs, suffix):
    golden = json.loads((BASELINE_DIR / f"demo_clip_{suffix}").read_text())
    assert suffix in current_outputs, f"no {suffix} written"
    rel, abs_ = TOLERANCE[suffix]
    _assert_within_tolerance(json.loads(current_outputs[suffix]), golden, rel, abs_)


class TestToleranceComparisonHelper:
    """Always-on coverage of the comparison itself; needs no models."""

    def test_identical_structures_pass(self):
        data = {"a": 1, "b": [1.0, 2.0], "c": "text"}
        _assert_within_tolerance(data, data, rel=1e-6)

    def test_float_within_tolerance_passes(self):
        _assert_within_tolerance({"score": 0.700049}, {"score": 0.7}, rel=1e-3)

    def test_small_value_within_absolute_tolerance_passes(self):
        _assert_within_tolerance({"au": 0.0031}, {"au": 0.0012}, rel=1e-2, abs_=1e-2)

    def test_float_outside_tolerance_fails(self):
        with pytest.raises(AssertionError):
            _assert_within_tolerance({"score": 0.9}, {"score": 0.7}, rel=1e-3)

    def test_non_float_mismatch_fails_exactly(self):
        with pytest.raises(AssertionError):
            _assert_within_tolerance({"label": "dog"}, {"label": "cat"}, rel=1e-3)

    def test_missing_key_fails(self):
        with pytest.raises(AssertionError):
            _assert_within_tolerance({"a": 1}, {"a": 1, "b": 2}, rel=1e-3)
