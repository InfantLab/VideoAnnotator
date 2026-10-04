"""The job runner records provenance in every file a pipeline writes (spec 017)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.provenance import ModelRef, companion_path, read_record
from videoannotator.storage.base import StorageBackend


class _Fake:
    """Writes one file the way the real pipeline of `name` names it."""

    suffix = ""
    content = ""

    def __init__(self, config):
        self.config = {"threshold": 0.5, "huggingface_token": "hf_secret", **config}

    def initialize(self):
        pass

    def provenance_models(self):
        return [ModelRef("m", "file", "abc", "sha256")]

    def process(self, video_path, start_time, end_time, pps, output_dir):
        stem = Path(video_path).stem
        (Path(output_dir) / f"{stem}_{self.suffix}").write_text(self.content)
        return [{}]

    def cleanup(self):
        pass


class _Scene(_Fake):
    suffix = "scene_detection.json"
    content = json.dumps({"images": [], "annotations": [], "categories": []})


class _Speech(_Fake):
    suffix = "speech_recognition.vtt"
    content = "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHi\n\n"


class _Diarization(_Fake):
    suffix = "speaker_diarization.rttm"
    content = "SPEAKER clip 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"


def _storage():
    storage = MagicMock(spec=StorageBackend)
    storage.annotation_exists.return_value = False
    storage.load_job_metadata.return_value = None
    storage.save_annotations.return_value = "database:/annotations/x"
    return storage


def test_every_output_is_stamped_and_the_job_keeps_the_record(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video bytes")
    job = run_job_pipelines(
        BatchJob(
            video_path=video,
            output_dir=tmp_path,
            selected_pipelines=["scene_detection", "speech_recognition", "speaker_diarization"],
            config={"deterministic": True},
        ),
        _storage(),
        {"scene_detection": _Scene, "speech_recognition": _Speech, "speaker_diarization": _Diarization},
    )  # fmt: skip
    assert job.status == JobStatus.COMPLETED

    for name, suffix in [
        ("scene_detection", "scene_detection.json"),
        ("speech_recognition", "speech_recognition.vtt"),
        ("speaker_diarization", "speaker_diarization.rttm"),
    ]:
        record = read_record(tmp_path / f"clip_{suffix}")
        assert record is not None, suffix
        assert record["pipeline"]["name"] == name
        assert record["job_id"] == job.job_id
        assert record["models"][0]["revision"] == "abc"
        assert record["settings"]["huggingface_token"] == "<redacted>"
        assert record["input"]["name"] == "clip.mp4"
        assert len(record["input"]["sha256"]) == 64
        # Empty only where torch is absent (core install).
        assert record["determinism"] == {} or record["determinism"]["deterministic"]
        assert job.pipeline_results[name].provenance == record

    assert companion_path(tmp_path / "clip_speaker_diarization.rttm").is_file()
    assert (
        tmp_path / "clip_speaker_diarization.rttm"
    ).read_text() == _Diarization.content


def test_a_failed_pipeline_keeps_what_was_known(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    class _Broken(_Scene):
        def process(self, **kwargs):
            raise RuntimeError("boom")

    job = run_job_pipelines(
        BatchJob(video_path=video, output_dir=tmp_path, selected_pipelines=["scene_detection"]),
        _storage(),
        {"scene_detection": _Broken},
    )  # fmt: skip
    result = job.pipeline_results["scene_detection"]
    assert result.status == JobStatus.FAILED
    assert result.provenance["models"][0]["name"] == "m"


def test_a_vlm_job_records_its_prompt_in_the_library(tmp_path):
    from unittest.mock import patch as mock_patch

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    class _Vlm(_Fake):
        suffix = "vlm_annotation.json"
        content = json.dumps({"annotations": []})

        def __init__(self, config):
            self.config = {"prompt": "Touch?", "model": "gemma4:e4b"}

        def provenance_vlm(self):
            return {
                "prompt_sha256": "x",
                "model_digest": None,
                "quantization": None,
                "base_url": "u",
            }

    with mock_patch("videoannotator.batch.job_execution.record_use_quietly") as record:
        job = run_job_pipelines(
            BatchJob(video_path=video, output_dir=tmp_path, selected_pipelines=["vlm_annotation"]),
            _storage(),
            {"vlm_annotation": _Vlm},
        )  # fmt: skip
    record.assert_called_once_with("Touch?", "gemma4:e4b", "job", job_id=job.job_id)
