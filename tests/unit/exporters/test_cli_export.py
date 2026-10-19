"""`videoannotator export ...` on a results folder, no server (spec 025)."""

from __future__ import annotations

import builtins
import csv
import io
import json

import pytest
from typer.testing import CliRunner

from videoannotator.cli import app
from videoannotator.exporters.readers import COLUMNS

from .conftest import RUN_NAME

runner = CliRunner()


def test_table_defaults_to_a_csv_named_after_the_folder(run_folder):
    result = runner.invoke(app, ["export", "table", str(run_folder)])
    assert result.exit_code == 0, result.output
    path = run_folder / f"{RUN_NAME}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert rows and list(rows[0]) == list(COLUMNS)
    assert "rows, 1 video, 7 pipelines" in result.output


def test_table_again_replaces_its_own_export_and_reads_nothing_new(run_folder):
    runner.invoke(app, ["export", "table", str(run_folder)])
    first = (run_folder / f"{RUN_NAME}.csv").read_text()
    result = runner.invoke(app, ["export", "table", str(run_folder)])
    assert result.exit_code == 0
    assert (run_folder / f"{RUN_NAME}.csv").read_text() == first


def test_table_to_stdout(run_folder):
    result = runner.invoke(app, ["export", "table", str(run_folder), "-o", "-"])
    assert result.exit_code == 0
    assert result.stdout.splitlines()[0] == ",".join(COLUMNS)


def test_json_summary(run_folder, tmp_path):
    out = tmp_path / "t.csv"
    result = runner.invoke(
        app, ["export", "table", str(run_folder), "-o", str(out), "--json"]
    )
    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["output"] == str(out)
    assert summary["videos"] == 1 and summary["table_format_version"] == 2


def test_refuses_to_overwrite_a_file_that_isnt_an_export(run_folder, tmp_path):
    precious = tmp_path / "notes.txt"
    precious.write_text("mine")
    result = runner.invoke(
        app, ["export", "table", str(run_folder), "-o", str(precious)]
    )
    assert result.exit_code == 1
    assert precious.read_text() == "mine"


def test_nothing_to_export(tmp_path):
    result = runner.invoke(app, ["export", "table", str(tmp_path)])
    assert result.exit_code == 1
    assert "No VideoAnnotator results" in result.output


def test_missing_folder(tmp_path):
    result = runner.invoke(app, ["export", "table", str(tmp_path / "nope")])
    assert result.exit_code == 1


def test_bad_format_is_a_usage_error(run_folder):
    result = runner.invoke(
        app, ["export", "table", str(run_folder), "--format", "xlsx"]
    )
    assert result.exit_code == 2


def test_parquet(run_folder):
    pytest.importorskip("pyarrow")
    result = runner.invoke(
        app, ["export", "table", str(run_folder), "--format", "parquet"]
    )
    assert result.exit_code == 0, result.output
    assert (run_folder / f"{RUN_NAME}.parquet").stat().st_size > 0


def test_parquet_not_installed_exits_3_with_the_install_command(
    run_folder, monkeypatch
):
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    result = runner.invoke(
        app, ["export", "table", str(run_folder), "--format", "parquet"]
    )
    assert result.exit_code == 3
    assert "videoannotator[parquet]" in result.output


def test_skipped_files_are_reported(run_folder):
    (run_folder / "demo_clip" / "notes.txt").write_text("x")
    result = runner.invoke(app, ["export", "table", str(run_folder), "-o", "-"])
    assert result.exit_code == 0
    assert "skipped notes.txt (demo_clip): no reader for this file" in result.stderr
    assert "notes.txt" not in io.StringIO(result.stdout).read()


def test_methods_prints_markdown(run_folder):
    result = runner.invoke(app, ["export", "methods", str(run_folder)])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# Methods")
    assert "@software{addyman2026videoannotator," in result.stdout


def test_methods_to_a_file_and_as_json(run_folder, tmp_path):
    out = tmp_path / "methods.md"
    result = runner.invoke(app, ["export", "methods", str(run_folder), "-o", str(out)])
    assert result.exit_code == 0
    assert out.read_text().startswith("# Methods")
    result = runner.invoke(app, ["export", "methods", str(run_folder), "--json"])
    body = json.loads(result.stdout)
    assert set(body) == {"prose", "bibtex", "revisions", "prompts", "summary"}


def test_methods_with_nothing_to_describe(tmp_path):
    result = runner.invoke(app, ["export", "methods", str(tmp_path)])
    assert result.exit_code == 1


def test_elan_beside_each_videos_results(run_folder):
    result = runner.invoke(app, ["export", "elan", str(run_folder)])
    assert result.exit_code == 0, result.output
    assert (run_folder / "demo_clip" / "demo_clip.eaf").is_file()
    assert "1 ELAN file" in result.output
    assert "per-detection measures" in result.stderr


def test_elan_to_a_folder_by_run(run_folder, tmp_path):
    out = tmp_path / "elan"
    result = runner.invoke(app, ["export", "elan", str(run_folder), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / RUN_NAME / "demo_clip.eaf").is_file()
