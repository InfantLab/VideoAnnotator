"""Naming and layout rules of the results folder (spec 022, research.md R2/R3)."""

import json
import os
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from videoannotator import results_folder as rf
from videoannotator.batch.types import BatchJob, JobStatus


class TestSanitize:
    @pytest.mark.parametrize("char", list('<>:"/\\|?*') + ["\x00", "\x1f"])
    def test_drops_characters_windows_forbids(self, char):
        assert rf.sanitize_component(f"a{char}b") == "ab"

    def test_strips_trailing_dots_and_spaces(self):
        assert rf.sanitize_component("wave 2. . ") == "wave 2"

    @pytest.mark.parametrize("name", ["CON", "nul", "com3", "LPT9", "nul.txt"])
    def test_avoids_reserved_device_names(self, name):
        assert rf.sanitize_component(name) == "_" + name

    def test_caps_length(self):
        assert len(rf.sanitize_component("x" * 300)) == rf.MAX_COMPONENT

    def test_keeps_unicode(self):
        assert rf.sanitize_component("Bébé 👶 study") == "Bébé 👶 study"

    def test_empty_falls_back(self):
        assert rf.sanitize_component("???", "video") == "video"


class TestRunName:
    def test_first_set_wins(self):
        assert rf.run_name(batch_name="Wave 2", dataset_name="DS") == "Wave 2"
        assert rf.run_name(dataset_name="DS", source_folder=Path("/x/F")) == "DS"
        assert rf.run_name(source_folder=Path("/x/F"), video=Path("v.mp4")) == "F"
        assert rf.run_name(video=Path("/x/child01.mp4")) == "child01"

    def test_falls_back_to_date_and_time(self):
        now = datetime(2026, 10, 6, 9, 5)
        assert rf.run_name(batch_name="  ", now=now) == "Run 2026-10-06 09-05"


def _run(name="BabyJokes wave 2", **kwargs):
    return rf.RunFolder.create(
        name,
        batch_id="b1",
        pipelines=["scene_detection"],
        config={},
        today=date(2026, 10, 6),
        **kwargs,
    )


class TestRunFolder:
    def test_named_after_run_and_date(self, results_root):
        run = _run()
        assert run.path == results_root / "BabyJokes wave 2 (2026-10-06)"
        assert (run.path / "run.json").is_file()

    def test_same_name_same_day_is_numbered_not_overwritten(self, results_root):
        first = _run()
        (first.path / "keep.txt").write_text("results")
        second = _run()
        third = _run()
        assert second.path.name == "BabyJokes wave 2 (2026-10-06 2)"
        assert third.path.name == "BabyJokes wave 2 (2026-10-06 3)"
        assert (first.path / "keep.txt").read_text() == "results"

    def test_long_names_stay_within_the_cap(self, results_root):
        assert len(_run("y" * 300).path.name) <= rf.MAX_COMPONENT

    def test_unwritable_root_names_the_folder(self, tmp_path, monkeypatch):
        blocker = tmp_path / "file"
        blocker.write_text("not a folder")
        monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_DIR", str(blocker / "results"))
        with pytest.raises(rf.ResultsFolderError) as e:
            _run()
        assert str(blocker / "results") in str(e.value)

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,
        reason="permission bits don't stop Windows or root",
    )
    def test_read_only_root_is_refused(self, results_root):
        results_root.mkdir(parents=True)
        results_root.chmod(0o555)
        try:
            with pytest.raises(rf.ResultsFolderError):
                rf.check_writable()
        finally:
            results_root.chmod(0o755)


class TestVideoFolders:
    def _job(self):
        return BatchJob(status=JobStatus.PENDING)

    def test_named_after_the_video_and_set_as_output_dir(self, results_root):
        run = _run()
        job = self._job()
        folder = run.add_video(job, Path("/data/child01.mp4"), size_bytes=3)
        assert folder == run.path / "child01"
        assert job.output_dir == folder

    def test_same_stem_uses_the_path_within_the_source(self, results_root, tmp_path):
        source = tmp_path / "study"
        for sub in ("site_a", "site_b"):
            (source / sub).mkdir(parents=True)
            (source / sub / "child01.mp4").write_bytes(b"v")
        run = _run(source_root=source)
        first = run.add_video(self._job(), source / "site_a" / "child01.mp4")
        second = run.add_video(self._job(), source / "site_b" / "child01.mp4")
        assert first.name == "child01"
        assert second.name == "site_b__child01"

    def test_still_colliding_names_are_numbered(self, results_root):
        run = _run()
        names = [run.add_video(self._job(), Path("/x/a.mp4")).name for _ in range(3)]
        assert names == ["a", "a 2", "a 3"]


class TestDisplayPath:
    def test_maps_container_prefixes_to_host(self, monkeypatch):
        monkeypatch.setenv(
            "VIDEOANNOTATOR_HOST_PATHS",
            "/results=/home/ada/VideoAnnotator;/videos=/home/ada/Studies;"
            "/videos/deep=/mnt/deep",
        )
        assert (
            rf.display_path("/results/Run (x)/a")
            == "/home/ada/VideoAnnotator/Run (x)/a"
        )
        assert rf.display_path("/videos/deep/b.mp4") == "/mnt/deep/b.mp4"
        assert rf.display_path("/videos") == "/home/ada/Studies"
        assert rf.display_path("/videosX/c.mp4") == "/videosX/c.mp4"

    def test_unchanged_without_mapping(self):
        assert rf.display_path("/home/ada/x") == "/home/ada/x"


class TestOverlap:
    def test_results_inside_a_configured_video_folder(self, results_root):
        assert rf.results_root_overlaps([results_root.parent]) == results_root.parent

    def test_separate_folders_do_not_overlap(self, results_root, tmp_path):
        assert rf.results_root_overlaps([tmp_path / "elsewhere"]) is None


class TestRunRecord:
    def test_shape(self, results_root):
        run = _run()
        job = BatchJob(status=JobStatus.PENDING)
        run.add_video(job, Path("/home/ada/child01.mp4"), size_bytes=499675)
        upload = BatchJob(status=JobStatus.PENDING)
        run.add_video(
            upload, Path("/internal/jobs/x/child02.mp4"), uploaded_name="child02.mp4"
        )
        record = rf.read_record(run.path)
        assert record["format"] == "videoannotator-run"
        assert record["format_version"] == 1
        assert record["run"]["name"] == "BabyJokes wave 2"
        assert record["pipelines"] == ["scene_detection"]
        assert record["videos"][0]["source"] == {
            "kind": "in_place",
            "path": "/home/ada/child01.mp4",
            "size_bytes": 499675,
        }
        uploaded = record["videos"][1]["source"]
        assert uploaded == {"kind": "uploaded", "original_filename": "child02.mp4"}
        assert "/internal" not in json.dumps(uploaded)

    def test_config_secrets_are_redacted(self, results_root):
        run = rf.RunFolder.create(
            "R",
            batch_id=None,
            pipelines=[],
            config={"vlm": {"api_key": "sk-secret"}},
        )
        assert "sk-secret" not in (run.path / "run.json").read_text()

    def test_finished_job_is_recorded(self, results_root):
        run = _run()
        job = BatchJob(status=JobStatus.PENDING)
        folder = run.add_video(job, Path("/x/child01.mp4"), size_bytes=1)
        (folder / "child01_scene_detection.json").write_text("{}")
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime(2026, 10, 6, 10, 1, 12)
        job.pipeline_results = {
            "scene_detection": SimpleNamespace(
                provenance={"models": [{"name": "clip", "revision": "abc"}]}
            )
        }
        rf.record_job_finished(job)
        entry = rf.read_record(run.path)["videos"][0]
        assert entry["status"] == "completed"
        assert entry["files"] == ["child01_scene_detection.json"]
        assert entry["models"] == {
            "scene_detection": [{"name": "clip", "revision": "abc"}]
        }
        assert entry["finished_at"]

    def test_jobs_from_before_the_feature_are_ignored(self):
        rf.record_job_finished(BatchJob(status=JobStatus.COMPLETED))

    def test_concurrent_updates_keep_every_entry(self, results_root):
        run = _run()
        jobs = [BatchJob(status=JobStatus.PENDING) for _ in range(20)]
        threads = [
            threading.Thread(
                target=run.add_video,
                args=(job, Path(f"/x/v{i}.mp4")),
                kwargs={"size_bytes": 1},
            )
            for i, job in enumerate(jobs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        record = rf.read_record(run.path)
        assert len(record["videos"]) == 20
        assert not [p for p in run.path.iterdir() if p.name.startswith(".run-")]


class TestRemoveJobResults:
    def test_removes_video_folder_then_empty_run_folder(self, results_root, tmp_path):
        video = tmp_path / "child01.mp4"
        video.write_bytes(b"original")
        run = _run()
        a, b = (
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.COMPLETED),
        )
        run.add_video(a, video)
        run.add_video(b, tmp_path / "child02.mp4", size_bytes=1)
        rf.remove_job_results(a)
        assert not Path(a.output_dir).exists()
        assert [v["job_id"] for v in rf.read_record(run.path)["videos"]] == [b.job_id]
        rf.remove_job_results(b)
        assert not run.path.exists()
        assert video.read_bytes() == b"original"

    def test_never_removes_anything_outside_the_results_root(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        job = BatchJob(status=JobStatus.COMPLETED, output_dir=outside)
        rf.remove_job_results(job)
        assert outside.is_dir()
