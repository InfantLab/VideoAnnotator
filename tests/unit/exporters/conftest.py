"""Results folders built from the committed demo-clip outputs (spec 025)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "viewer_contract"
LEGACY = FIXTURES / "legacy"
RUN_NAME = "Demo run (2026-10-16)"


def copy_outputs(source: Path, dest: Path) -> Path:
    """Copy the demo clip's output files (not READMEs or subfolders) into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name.startswith("demo_clip_"):
            shutil.copy2(path, dest / path.name)
    return dest


def make_run(root: Path, videos: dict[str, Path], name: str = RUN_NAME) -> Path:
    """A spec 022 run folder: `run.json` plus one folder per video."""
    run = root / name
    run.mkdir(parents=True)
    entries = []
    for video, source in videos.items():
        copy_outputs(source, run / video)
        entries.append(
            {
                "job_id": f"job-{video}",
                "folder": video,
                "source": {"kind": "in_place", "path": f"/data/videos/{video}.mp4"},
                "status": "completed",
            }
        )
    record = {
        "format": "videoannotator-run",
        "format_version": 1,
        "run": {"batch_id": "batch-1", "name": name.split(" (")[0]},
        "videoannotator_version": "1.6.0",
        "pipelines": [],
        "config": {},
        "videos": entries,
    }
    (run / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    """A run with one video, `demo_clip`, holding the current outputs."""
    return make_run(tmp_path, {"demo_clip": FIXTURES})


@pytest.fixture
def legacy_folder(tmp_path: Path) -> Path:
    """A flat folder of outputs from before provenance (v1.5.0)."""
    return copy_outputs(LEGACY, tmp_path / "legacy job")


@pytest.fixture
def flat_folder(tmp_path: Path) -> Path:
    """A pre-022 job folder: outputs, the video, and the storage backend's files."""
    folder = copy_outputs(FIXTURES, tmp_path / "1b47a872-job")
    (folder / "demo_clip.mp4").write_bytes(b"\x00")
    (folder / "job_metadata.json").write_text("{}", encoding="utf-8")
    (folder / "person_tracking.json").write_text("{}", encoding="utf-8")
    return folder
