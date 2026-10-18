"""scripts/compare_demo_outputs.py tells identical, noisy and changed runs apart."""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "compare_demo_outputs",
    Path(__file__).resolve().parents[2] / "scripts" / "compare_demo_outputs.py",
)
assert _spec and _spec.loader
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

VTT = """WEBVTT

NOTE videoannotator-provenance {{"run": "{run}"}}

1
00:00:00.000 --> 00:00:01.320
Ready, baby girl?

2
00:00:02.000 --> 00:00:03.160
{second}
"""


def _write(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir()
    for name, text in files.items():
        (folder / f"demo_clip_{name}").write_text(text)
    return folder


def _coco(score: float, provenance: str, keys: dict | None = None) -> str:
    ann = {"id": 1, "image_id": 1, "score": score, "track_id": 3}
    ann.update(keys or {})
    return json.dumps(
        {
            "info": {"date_created": provenance},
            "images": [{"id": 1}],
            "annotations": [ann],
            "provenance": {"created_at": provenance},
        }
    )


def test_provenance_and_dates_are_ignored(tmp_path):
    before = _write(
        tmp_path / "a",
        {
            "speech_recognition.vtt": VTT.format(run="1", second="Good morning."),
            "scene_detection.json": _coco(0.5, "monday"),
        },
    )
    after = _write(
        tmp_path / "b",
        {
            "speech_recognition.vtt": VTT.format(run="2", second="Good morning."),
            "scene_detection.json": _coco(0.5, "tuesday"),
        },
    )
    assert cmp.compare(before, after) == [
        "scene_detection.json: identical",
        "speech_recognition.vtt: identical",
    ]


def test_noise_within_tolerance_is_reported_as_such(tmp_path):
    before = _write(tmp_path / "a", {"person_tracking.json": _coco(0.7, "x")})
    after = _write(tmp_path / "b", {"person_tracking.json": _coco(0.70001, "y")})
    assert cmp.compare(before, after)[0].startswith(
        "person_tracking.json: within tolerance"
    )


def test_a_reworded_cue_is_counted_and_shown(tmp_path):
    before = _write(
        tmp_path / "a",
        {"speech_recognition.vtt": VTT.format(run="1", second="Good morning.")},
    )
    after = _write(
        tmp_path / "b",
        {"speech_recognition.vtt": VTT.format(run="1", second="Good morning, girl.")},
    )
    report = cmp.compare(before, after)
    assert report[0] == "speech_recognition.vtt: different"
    assert "cues 2 -> 2: 1 reworded, 0 added, 0 removed" in report[1]
    assert "Good morning, girl." in "\n".join(report)


def test_speaker_turns_are_summarised(tmp_path):
    line = "SPEAKER video 1 {start} 1.0 <NA> <NA> {spk} <NA> <NA>"
    before = _write(
        tmp_path / "a",
        {"speaker_diarization.rttm": line.format(start=0.5, spk="SPEAKER_00")},
    )
    after = _write(
        tmp_path / "b",
        {
            "speaker_diarization.rttm": "\n".join(
                [
                    line.format(start=0.5, spk="SPEAKER_00"),
                    line.format(start=2.0, spk="SPEAKER_01"),
                ]
            )
        },
    )
    report = cmp.compare(before, after)
    assert "before: 1 turns, 1 speakers" in report[1]
    assert "after: 2 turns, 2 speakers" in report[2]


def test_dropped_attribute_keys_show_in_the_summary(tmp_path):
    before = _write(
        tmp_path / "a",
        {"face_detections.json": _coco(0.9, "x", {"age": 30.0, "emotion": "happy"})},
    )
    after = _write(
        tmp_path / "b",
        {"face_detections.json": _coco(0.9, "x", {"emotion": "happy"})},
    )
    report = "\n".join(cmp.compare(before, after))
    assert "face_detections.json: different" in report
    assert "keys: age, emotion" in report
    assert "keys: emotion, id" in report


def test_a_file_missing_on_one_side_is_named(tmp_path):
    before = _write(tmp_path / "a", {"speaker_diarization.rttm": ""})
    after = _write(tmp_path / "b", {})
    assert cmp.compare(before, after) == ["speaker_diarization.rttm: only in before"]
