"""The results table: columns, order, CSV/Parquet, summary (spec 025, US1)."""

from __future__ import annotations

import builtins
import csv
import io
import json
from pathlib import Path

import pytest

from videoannotator.exporters.readers import COLUMNS, NOT_RECORDED
from videoannotator.exporters.results_set import from_folder
from videoannotator.exporters.table import (
    ExportSummary,
    ParquetNotInstalled,
    parquet_available,
    rows,
    write_csv,
    write_parquet,
)

from .conftest import FIXTURES, RUN_NAME


def _csv(folder: Path) -> tuple[list[dict[str, str]], ExportSummary]:
    out = io.StringIO(newline="")
    summary = write_csv(from_folder(folder), out)
    return list(csv.DictReader(io.StringIO(out.getvalue()))), summary


def test_columns_are_the_contract(run_folder):
    out = io.StringIO(newline="")
    write_csv(from_folder(run_folder), out)
    assert out.getvalue().splitlines()[0] == ",".join(COLUMNS)


def test_every_pipeline_appears_with_run_and_video(run_folder):
    table, summary = _csv(run_folder)
    assert {r["run"] for r in table} == {RUN_NAME}
    assert {r["video"] for r in table} == {"demo_clip"}
    assert summary.pipelines == sorted(
        {
            "person_tracking",
            "face_analysis",
            "face_openface3_embedding",
            "scene_detection",
            "vlm_annotation",
            "speaker_diarization",
            "speech_recognition",
        }
    )
    assert summary.rows == len(table)
    assert summary.videos == 1


def test_measure_counts_match_the_native_files(run_folder):
    """SC-002: every native event is in the table, nothing invented."""
    table, _ = _csv(run_folder)

    def count(measure: str) -> int:
        return sum(r["measure"] == measure for r in table)

    def coco(name: str) -> list[dict]:
        return json.loads((FIXTURES / f"demo_clip_{name}.json").read_text())[
            "annotations"
        ]

    assert count("person") == len(coco("person_tracking"))
    assert count("face") == len(coco("face_detections")) + len(
        coco("openface3_analysis")
    )
    assert count("scene") == len(coco("scene_detection"))
    assert count("vlm_label") == len(coco("vlm_annotation"))
    assert count("speaker_turn") == 4
    vtt = (FIXTURES / "demo_clip_speech_recognition.vtt").read_text()
    assert count("speech") == vtt.count("-->")
    assert count("action_unit") == sum(
        len(a["openface3"]["action_units"]) for a in coco("openface3_analysis")
    )


def test_numbers_round_trip_exactly(run_folder):
    table, _ = _csv(run_folder)
    native = json.loads((FIXTURES / "demo_clip_person_tracking.json").read_text())
    first = native["annotations"][0]
    row = next(r for r in table if r["measure"] == "person" and r["frame"] == "0")
    assert float(row["confidence"]) == first["score"]
    assert [float(row[k]) for k in ("x", "y", "width", "height")] == first["bbox"]
    assert row["end"] == ""  # a moment, not an interval
    assert row["track"] == first["person_id"]
    assert row["model"] == "models/yolo/yolo11n-pose.pt"
    assert row["videoannotator_version"] == "1.5.0"
    assert row["file"] == "demo_clip_person_tracking.json"


def test_rows_are_ordered_by_video_then_time(run_folder):
    table, _ = _csv(run_folder)
    starts = [float(r["start"]) for r in table]
    assert starts == sorted(starts)
    # rows of one event stay together, in measure order
    face = [
        r
        for r in table
        if r["pipeline"] == "face_openface3_embedding" and r["event"] == "1"
    ]
    assert face[0]["measure"] == "face"
    assert [r["measure"] for r in face[1:9]] == ["action_unit"] * 8


def test_events_are_numbered_per_pipeline(run_folder):
    table, _ = _csv(run_folder)
    turns = [r for r in table if r["pipeline"] == "speaker_diarization"]
    assert [r["event"] for r in turns] == ["1", "2", "3", "4"]


def test_export_is_identical_when_repeated(run_folder):
    first, _ = _csv(run_folder)
    second, _ = _csv(run_folder)
    assert first == second


def test_files_without_provenance_say_not_recorded(legacy_folder):
    table, _ = _csv(legacy_folder)
    assert table
    non_vlm = [r for r in table if r["pipeline"] != "vlm_annotation"]
    assert {r["model"] for r in non_vlm} == {NOT_RECORDED}
    assert {r["videoannotator_version"] for r in table} == {NOT_RECORDED}
    # VLM annotations name their model themselves.
    assert {r["model"] for r in table if r["pipeline"] == "vlm_annotation"} == {
        "qwen3.5:9b"
    }


def test_awkward_text_survives_csv(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    text = 'She said "hi, baby",\nthen 你好 — ok'
    (folder / "clip_speech_recognition.vtt").write_text(
        f"WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n{text}\n", encoding="utf-8"
    )
    out = io.StringIO(newline="")
    write_csv(from_folder(folder), out)
    [row] = list(csv.DictReader(io.StringIO(out.getvalue())))
    assert row["label"] == text
    pandas = pytest.importorskip("pandas")
    frame = pandas.read_csv(io.StringIO(out.getvalue()))
    assert frame.loc[0, "label"] == text
    assert len(frame) == 1


def test_unreadable_file_is_skipped_and_the_rest_exports(run_folder):
    (run_folder / "demo_clip" / "demo_clip_scene_detection.json").write_text("{broken")
    table, summary = _csv(run_folder)
    assert not any(r["measure"] == "scene" for r in table)
    assert any(r["measure"] == "speech" for r in table)
    [skipped] = summary.skipped
    assert skipped["file"] == "demo_clip_scene_detection.json"
    assert skipped["reason"].startswith("unreadable")


def test_summary_header_is_capped(run_folder):
    summary = ExportSummary(
        missing=[{"run": "r", "video": f"v{i}", "reason": "x" * 50} for i in range(500)]
    )
    header = summary.to_header(max_bytes=8192)
    assert len(header) <= 8192
    data = json.loads(header)
    assert data["truncated"] is True
    assert header.isascii()


def test_sources_hold_each_files_provenance(run_folder):
    summary = ExportSummary()
    list(rows(from_folder(run_folder), summary))
    by_file = {s["file"]: s["provenance"] for s in summary.sources}
    assert by_file["demo_clip_speaker_diarization.rttm"]["pipeline"]["name"] == (
        "speaker_diarization"
    )
    assert by_file["demo_clip_face_detections.json"] is None


@pytest.mark.skipif(not parquet_available(), reason="pyarrow not installed")
def test_parquet_has_the_same_rows_typed(run_folder, tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "t.parquet"
    summary = write_parquet(from_folder(run_folder), path)
    table = pq.read_table(path)
    assert table.column_names == list(COLUMNS)
    assert table.num_rows == summary.rows
    assert str(table.schema.field("start").type) == "double"
    assert str(table.schema.field("event").type) == "int64"
    csv_rows, _ = _csv(run_folder)
    first = table.slice(0, 1).to_pylist()[0]
    assert first["start"] == float(csv_rows[0]["start"])
    meta = pq.read_metadata(path).metadata
    assert meta[b"videoannotator.table_format_version"] == b"1"
    stored = json.loads(meta[b"videoannotator.export_summary"])
    assert stored["rows"] == summary.rows and stored["sources"]


def test_parquet_without_pyarrow_says_how_to_install(run_folder, tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    with pytest.raises(ParquetNotInstalled, match=r"videoannotator\[parquet\]"):
        write_parquet(from_folder(run_folder), tmp_path / "t.parquet")
