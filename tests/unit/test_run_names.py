"""A run's name says what it ran on and what it found, without a date (UAT #41)."""

import json
from pathlib import Path
from types import SimpleNamespace

from videoannotator.api.v1.video_list import _default_name
from videoannotator.results_folder import RUN_RECORD
from videoannotator.video_index import _run_name


class _Index:
    def video_for(self, path):
        return None


def test_default_name_is_the_subject_then_what_was_found():
    one = [Path("/data/pk/expCP2.mov")]
    assert _default_name(one, None, _Index(), ["scene_detection"]) == (
        "expCP2 · PySceneDetect + CLIP"
    )
    folder = [Path("/data/pk/a.mp4"), Path("/data/pk/b.mp4")]
    assert _default_name(
        folder, None, _Index(), ["speaker_diarization", "speech_recognition"]
    ) == ("pk · pyannote community-1, Whisper large-v3-turbo")
    mixed = [Path("/x/a.mp4"), Path("/y/b.mp4")]
    assert (
        _default_name(mixed, None, _Index(), ["a", "b", "c"])
        == "2 videos · 3 pipelines"
    )
    assert _default_name(one, SimpleNamespace(name="QS"), _Index()) == "QS"


def test_a_run_is_shown_by_its_given_name_not_its_dated_folder(tmp_path):
    folder = tmp_path / "expCP2 (2026-10-26 3)"
    (folder / "expCP2").mkdir(parents=True)
    (folder / RUN_RECORD).write_text(
        json.dumps({"run": {"name": "expCP2 · PySceneDetect + CLIP"}})
    )
    job = SimpleNamespace(
        batch_name=None, output_dir=str(folder / "expCP2"), video_path=None
    )
    assert _run_name([job]) == "expCP2 · PySceneDetect + CLIP"
