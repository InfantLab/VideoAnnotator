"""A value per finished result for the Videos table (UAT #10)."""

from __future__ import annotations

import json

from videoannotator.result_values import clear_cache, result_value

RTTM = (
    "SPEAKER clip 1 0.50 1.00 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    "SPEAKER clip 1 2.00 1.00 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    "SPEAKER clip 1 3.50 0.50 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
)
VTT = "WEBVTT\n\n00:00.500 --> 00:01.500\nPeekaboo there you are\n\n00:02.000 --> 00:03.000\nAgain\n"


def _faces(per_frame: list[int]) -> str:
    images = [
        {"id": i, "timestamp": i * 0.5, "frame_number": i}
        for i in range(len(per_frame))
    ]
    annotations = [
        {"id": f"{i}-{n}", "image_id": i, "bbox": [0, 0, 1, 1], "score": 0.9}
        for i, count in enumerate(per_frame)
        for n in range(count)
    ]
    return json.dumps({"images": images, "annotations": annotations})


def test_values_say_what_was_found(tmp_path):
    clear_cache()
    (tmp_path / "clip_speaker_diarization.rttm").write_text(RTTM)
    (tmp_path / "clip_speech_recognition.vtt").write_text(VTT)
    (tmp_path / "clip_face_detections.json").write_text(_faces([1, 3, 2]))

    speakers = result_value(tmp_path, "speaker_diarization")
    assert speakers is not None
    assert (speakers.count, speakers.text, speakers.detail) == (
        2,
        "2 speakers",
        "3 turns",
    )
    words = result_value(tmp_path, "speech_recognition")
    assert words is not None and (words.count, words.text) == (5, "5 words")
    faces = result_value(tmp_path, "face_analysis")
    assert faces is not None
    assert (faces.count, faces.text) == (3, "3 faces")
    assert faces.detail == "most in one frame; seen in 3 frames"


def test_nothing_found_and_no_file(tmp_path):
    clear_cache()
    (tmp_path / "clip_speech_recognition.vtt").write_text("WEBVTT\n")

    empty = result_value(tmp_path, "speech_recognition")
    assert empty is not None and (empty.count, empty.text) == (0, "no speech")
    assert result_value(tmp_path, "scene_detection") is None
    assert result_value(tmp_path / "gone", "scene_detection") is None


def test_a_changed_file_is_read_again(tmp_path):
    clear_cache()
    rttm = tmp_path / "clip_speaker_diarization.rttm"
    rttm.write_text(RTTM.splitlines(keepends=True)[0])
    first = result_value(tmp_path, "speaker_diarization")
    assert first is not None and first.count == 1
    rttm.write_text(RTTM)
    again = result_value(tmp_path, "speaker_diarization")
    assert again is not None and again.count == 2
