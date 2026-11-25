"""The BabyJokes tutorial's analysis scripts run on the demo's results table.

`examples/babyjokes/analysis.py` and `analysis.R` read the table a researcher
exports; here it is exported from the five demo clips bundled in the viewer,
with the dataset's rows for those clips (`tests/fixtures/babyjokes/`). An
export change that breaks the scripts fails here, not in the tutorial.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from videoannotator.exporters.results_set import from_folder
from videoannotator.exporters.table import write_csv

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "examples" / "babyjokes"
DEMO = ROOT / "viewer" / "public" / "demo" / "babyjokes"
DATA = ROOT / "tests" / "fixtures" / "babyjokes"
RATINGS = DATA / "babyjokes-public-ratings.csv"
CLIPS = DATA / "babyjokes-public-clips.csv"


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("babyjokes") / "BabyJokes.csv"
    with path.open("w", encoding="utf-8", newline="") as out:
        write_csv(from_folder(DEMO), out)
    return path


def _by_clip(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return {row["video"]: row for row in csv.DictReader(f)}


def _check(output: str, by_clip: dict[str, dict[str, str]]) -> None:
    assert "5 repetitions rated by a parent; 5 with the baby's face" in output
    assert "OpenFace 3, AU12: smiled in at least 25% of the baby's faces" in output
    assert "kappa" in output
    assert len(by_clip) == 5
    assert all(row["au12_share"] not in ("", "NA") for row in by_clip.values())


def test_python(results, tmp_path):
    out = tmp_path / "by_clip.csv"
    run = subprocess.run(
        [
            sys.executable,
            SCRIPTS / "analysis.py",
            results,
            RATINGS,
            CLIPS,
            "--out",
            out,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    _check(run.stdout, _by_clip(out))


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R is not installed")
def test_r_matches_python(results, tmp_path):
    py_out, r_out = tmp_path / "py.csv", tmp_path / "r.csv"
    subprocess.run(
        [
            sys.executable,
            SCRIPTS / "analysis.py",
            results,
            RATINGS,
            CLIPS,
            "--out",
            py_out,
        ],
        check=True,
        capture_output=True,
    )
    run = subprocess.run(
        ["Rscript", SCRIPTS / "analysis.R", results, RATINGS, CLIPS, r_out],
        capture_output=True,
        text=True,
        check=True,
    )
    r, py = _by_clip(r_out), _by_clip(py_out)
    _check(run.stdout, r)
    for video, row in py.items():
        for column in ("au12_share", "happy_share"):
            assert float(r[video][column]) == pytest.approx(float(row[column]))
