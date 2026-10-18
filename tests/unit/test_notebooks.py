"""The worked notebooks run, on a results table that is up to date (spec 025, US4)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "examples" / "notebooks"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_demo_table", REPO / "scripts" / "make_demo_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_table_is_a_fresh_export_of_the_fixtures(tmp_path):
    """Fails when the fixtures or the table format change: run
    `python scripts/make_demo_table.py` and commit the result."""
    from videoannotator.exporters.results_set import from_folder
    from videoannotator.exporters.table import write_csv

    script = _load_script()
    run = script.demo_run(tmp_path)
    fresh = tmp_path / "fresh.csv"
    with open(fresh, "w", encoding="utf-8", newline="") as out:
        write_csv(from_folder(run), out)
    assert (NOTEBOOKS / "demo_clip_table.csv").read_text(
        encoding="utf-8"
    ) == fresh.read_text(encoding="utf-8")


def test_python_notebook_runs_top_to_bottom(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    shutil.copy(NOTEBOOKS / "demo_clip_table.csv", tmp_path)
    monkeypatch.chdir(tmp_path)

    notebook = json.loads((NOTEBOOKS / "results_table_python.ipynb").read_text())
    namespace: dict = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "notebook", "exec"), namespace)

    df = namespace["df"]
    assert len(df) > 0
    assert not namespace["turns"].empty and not namespace["people"].empty


def test_r_notebook_reads_the_same_table():
    text = (NOTEBOOKS / "results_table_r.Rmd").read_text(encoding="utf-8")
    assert 'TABLE <- "demo_clip_table.csv"' in text


def test_notebooks_have_no_outputs_committed():
    notebook = json.loads((NOTEBOOKS / "results_table_python.ipynb").read_text())
    assert all(
        not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code"
    )


@pytest.mark.skipif(
    os.environ.get("CI_HAS_R") != "1", reason="R checked before release"
)
def test_r_notebook_runs(tmp_path):  # pragma: no cover - needs R (tests/manual)
    import subprocess

    for name in ("results_table_r.Rmd", "demo_clip_table.csv"):
        shutil.copy(NOTEBOOKS / name, tmp_path)
    subprocess.run(
        ["Rscript", "-e", 'rmarkdown::render("results_table_r.Rmd", quiet = TRUE)'],
        cwd=tmp_path,
        check=True,
    )
    assert (tmp_path / "results_table_r.html").is_file()
