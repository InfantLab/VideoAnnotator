"""Table format version 2 (spec 028 contracts/table-v2.md): `author`, note rows."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from videoannotator.exporters import table
from videoannotator.exporters.readers import COLUMNS, TABLE_FORMAT_VERSION
from videoannotator.exporters.results_set import ResultsSet, _collect

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "viewer_contract"
V1_COLUMNS = [
    "run",
    "video",
    "pipeline",
    "event",
    "track",
    "start",
    "end",
    "frame",
    "measure",
    "label",
    "value",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "model",
    "videoannotator_version",
    "file",
]


def _video(run: str, media: str) -> object:
    video = _collect(FIXTURES, run, "p01_t1")
    video.skipped.clear()
    video.media_path = media
    return video


def _note(start, text, end=None, author="Caspar", path="/data/p01_t1.mp4"):
    return SimpleNamespace(
        start_sec=start, end_sec=end, text=text, author_name=author, video_path=path
    )


NOTES = [_note(1.5, "infant looks away"), _note(4.0, "peekaboo", end=6.0)]


def _export(results, notes_for=None):
    out = io.StringIO(newline="")
    summary = table.write_csv(results, out, notes_for)
    return list(csv.DictReader(io.StringIO(out.getvalue()))), out.getvalue(), summary


def test_columns_one_to_nineteen_are_unchanged_and_author_is_last():
    assert list(COLUMNS[:19]) == V1_COLUMNS
    assert COLUMNS[-1] == "author"
    assert TABLE_FORMAT_VERSION == 2


def test_notes_come_once_per_video_however_many_runs():
    results = ResultsSet(
        videos=[
            _video("run A", "/data/p01_t1.mp4"),
            _video("run B", "/data/p01_t1.mp4"),
        ]
    )
    rows, _, summary = _export(results, lambda _video: NOTES)
    notes = [r for r in rows if r["pipeline"] == "notes"]
    assert len(notes) == 2
    assert summary.researcher_notes == 2
    first, second = notes
    assert first == {
        **{c: "" for c in COLUMNS},
        "video": "p01_t1",
        "pipeline": "notes",
        "event": "1",
        "start": "1.5",
        "measure": "note",
        "label": "infant looks away",
        "videoannotator_version": first["videoannotator_version"],
        "author": "Caspar",
    }
    assert first["videoannotator_version"]
    assert (second["event"], second["end"]) == ("2", "6.0")
    assert all(r["author"] == "" for r in rows if r["pipeline"] != "notes")
    assert "notes" in summary.pipelines


def test_no_notes_no_note_rows():
    results = ResultsSet(videos=[_video("run A", "/data/p01_t1.mp4")])
    rows, _, summary = _export(results, lambda _video: [])
    assert not [r for r in rows if r["pipeline"] == "notes"]
    assert summary.researcher_notes == 0


def test_an_export_without_the_database_says_notes_are_not_included():
    def unavailable(_video):
        raise table.NotesUnavailable()

    results = ResultsSet(videos=[_video("run A", "/data/p01_t1.mp4")])
    _, _, summary = _export(results, unavailable)
    assert table.NO_NOTES_DATABASE in summary.lines()
    assert summary.to_dict()["table_format_version"] == 2


def test_parquet_has_the_author_column_and_version_2(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    results = ResultsSet(videos=[_video("run A", "/data/p01_t1.mp4")])
    path = tmp_path / "t.parquet"
    summary = table.write_parquet(results, path, lambda _video: NOTES)
    read = pq.read_table(path)
    assert read.column_names == list(COLUMNS)
    meta = pq.read_metadata(path).metadata
    assert meta[b"videoannotator.table_format_version"] == b"2"
    assert summary.researcher_notes == 2
    authors = read.column("author").to_pylist()
    assert authors.count("Caspar") == 2
