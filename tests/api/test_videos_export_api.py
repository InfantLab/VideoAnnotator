"""POST /videos/export/*: a table, ELAN and methods for any videos (spec 029 R10)."""

from __future__ import annotations

import csv
import io
import json
import zipfile

from tests.fixtures.video_index import write_video
from tests.unit.exporters.conftest import FIXTURES, copy_outputs
from videoannotator.exporters.table import parquet_available

DONE = ("completed", None, None)


def _two_runs(api):
    """p01 ran twice: speaker turns only in the first, scenes in both."""
    video = write_video(api.shared / "pk" / "demo_clip.mp4")
    out = api.jobs.tmp_path / "out"
    older = api.jobs.make_job(
        video,
        batch_id="b1",
        batch_name="First",
        pipelines={"scene_detection": DONE, "speaker_diarization": DONE},
        output_dir=copy_outputs(FIXTURES, out / "first"),
    )
    newer = api.jobs.make_job(
        video,
        batch_id="b2",
        batch_name="Second",
        pipelines={"scene_detection": DONE},
        output_dir=copy_outputs(FIXTURES, out / "second"),
    )
    return video, older, newer


def _post(api, kind, **body):
    return api.client.post(f"/api/v1/videos/export/{kind}", json=body)


def _rows(response):
    return list(csv.DictReader(io.StringIO(response.text)))


def test_latest_takes_each_pipelines_newest_result(api):
    video, *_ = _two_runs(api)

    r = _post(api, "table", paths=[str(video)])

    assert r.status_code == 200, r.text
    runs = {(row["pipeline"], row["run"]) for row in _rows(r)}
    assert ("scene_detection", "Second") in runs
    assert ("scene_detection", "First") not in runs
    assert ("speaker_diarization", "First") in runs
    assert 'filename="demo_clip.csv"' in r.headers["content-disposition"]
    assert json.loads(r.headers["x-videoannotator-export-summary"])["videos"] >= 1


def test_all_takes_every_run(api):
    video, *_ = _two_runs(api)

    r = _post(api, "table", paths=[str(video)], results="all")

    runs = {(row["pipeline"], row["run"]) for row in _rows(r)}
    assert {("scene_detection", "First"), ("scene_detection", "Second")} <= runs


def test_a_project_and_a_run(api):
    video, *_ = _two_runs(api)
    project = api.project("Peekaboo", server_folder=str(api.shared / "pk"))

    by_project = _post(api, "table", project_id=project["id"])
    by_run = _post(api, "table", run_id="b1")

    assert by_project.status_code == 200, by_project.text
    assert 'filename="Peekaboo.csv"' in by_project.headers["content-disposition"]
    assert by_run.status_code == 200, by_run.text
    assert {row["run"] for row in _rows(by_run)} == {"First"}


def test_nothing_to_export(api):
    video = write_video(api.shared / "never_run.mp4")

    r = _post(api, "table", paths=[str(video)])

    assert (r.status_code, r.json()["error"]["code"]) == (409, "NOTHING_TO_EXPORT")


def test_exactly_one_set(api):
    r = _post(api, "table", paths=["/x"], run_id="b1")
    assert (r.status_code, r.json()["error"]["code"]) == (422, "INVALID_EXPORT_SCOPE")
    r = _post(api, "table", run_id="nope")
    assert (r.status_code, r.json()["error"]["code"]) == (404, "RUN_NOT_FOUND")


def test_parquet_needs_its_extra(api):
    video, *_ = _two_runs(api)

    r = _post(api, "table", paths=[str(video)], format="parquet")

    if parquet_available():
        assert r.status_code == 200
    else:
        assert r.json()["error"]["code"] == "PARQUET_NOT_INSTALLED"


def test_elan_and_methods(api):
    video, *_ = _two_runs(api)
    other = write_video(api.shared / "pk" / "second.mp4")
    api.jobs.make_job(
        other,
        pipelines={"scene_detection": DONE},
        output_dir=copy_outputs(FIXTURES, api.jobs.tmp_path / "out" / "other"),
    )

    one = _post(api, "elan", paths=[str(video)])
    both = _post(api, "elan", paths=[str(video), str(other)])
    methods = _post(api, "methods", paths=[str(video)])
    markdown = _post(api, "methods", paths=[str(video)], format="markdown")

    assert one.status_code == 200, one.text
    assert one.headers["content-disposition"].endswith(".eaf")
    assert both.status_code == 200, both.text
    assert len(zipfile.ZipFile(io.BytesIO(both.content)).namelist()) == 2
    assert methods.status_code == 200, methods.text
    assert methods.json()
    assert markdown.headers["content-type"].startswith("text/markdown")
