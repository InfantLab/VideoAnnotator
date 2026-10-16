"""Each native format reads into events that match the file exactly (SC-002)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videoannotator.exporters.readers import (
    ReaderError,
    classify,
    read_deepface,
    read_openface3,
    read_person,
    read_rttm,
    read_scene,
    read_vlm,
    read_webvtt,
)

from .conftest import FIXTURES, LEGACY


def _coco(name: str, folder: Path = FIXTURES) -> dict:
    return json.loads((folder / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("folder", [FIXTURES, LEGACY])
def test_person_tracking_one_event_per_detection(folder):
    native = _coco("demo_clip_person_tracking.json", folder)
    events = read_person(folder / "demo_clip_person_tracking.json").events
    assert len(events) == len(native["annotations"])
    for event, a in zip(events, native["annotations"], strict=True):
        assert event.start == a["timestamp"]
        assert event.frame == a["frame_number"]
        assert event.track == a["person_id"]
        assert event.box == tuple(a["bbox"])
        assert event.end is None
        [person] = event.measures
        assert (person.measure, person.label, person.confidence) == (
            "person",
            a["person_label"],
            a["score"],
        )


def test_deepface_face_emotion_age_gender():
    native = _coco("demo_clip_face_detections.json")
    events = read_deepface(FIXTURES / "demo_clip_face_detections.json").events
    assert len(events) == len(native["annotations"])
    a, event = native["annotations"][0], events[0]
    by_measure = {}
    for m in event.measures:
        by_measure.setdefault(m.measure, []).append(m)
    assert by_measure["face"][0].confidence == a["score"]
    assert (by_measure["emotion"][0].label, by_measure["emotion"][0].confidence) == (
        a["emotion"],
        a["emotion_confidence"],
    )
    assert {m.label: m.value for m in by_measure["emotion_score"]} == a[
        "emotion_scores"
    ]
    assert by_measure["age"][0].value == a["age"]
    assert by_measure["gender"][0].label == a["gender"]
    # face_id is an index within the frame, not a track.
    assert event.track == ""


def test_openface3_action_units_pose_gaze():
    native = _coco("demo_clip_openface3_analysis.json")
    events = read_openface3(FIXTURES / "demo_clip_openface3_analysis.json").events
    assert len(events) == len(native["annotations"])
    for event, a in zip(events, native["annotations"], strict=True):
        features = a["openface3"]
        assert event.start == features["timestamp"]
        assert event.box == tuple(a["bbox"])
        units = {m.label: m.value for m in event.measures if m.measure == "action_unit"}
        assert units == {k: v["intensity"] for k, v in features["action_units"].items()}
        presence = {
            m.label: m.value
            for m in event.measures
            if m.measure == "action_unit_presence"
        }
        assert presence == {
            k: int(v["presence"]) for k, v in features["action_units"].items()
        }
        pose = {m.label: m.value for m in event.measures if m.measure == "head_pose"}
        assert pose == {k: features["head_pose"][k] for k in ("pitch", "yaw", "roll")}
        gaze = [m for m in event.measures if m.measure == "gaze"]
        assert {m.label for m in gaze} == {"direction_x", "direction_y", "direction_z"}
        assert all(m.confidence == features["gaze"]["confidence"] for m in gaze)


def test_openface3_emotion_when_enabled(tmp_path):
    native = _coco("demo_clip_openface3_analysis.json")
    native["annotations"] = native["annotations"][:1]
    native["annotations"][0]["openface3"]["emotion"] = {
        "dominant": "happiness",
        "probabilities": {"neutral": 0.2, "happiness": 0.8},
        "valence": 0.6,
        "arousal": 0.3,
    }
    path = tmp_path / "x_openface3_analysis.json"
    path.write_text(json.dumps(native))
    [event] = read_openface3(path).events
    rows = [(m.measure, m.label, m.value, m.confidence) for m in event.measures]
    assert ("emotion", "happiness", None, 0.8) in rows
    assert ("emotion_score", "neutral", 0.2, None) in rows
    assert ("valence", None, 0.6, None) in rows
    assert ("arousal", None, 0.3, None) in rows


def test_scene_interval_and_scores():
    native = _coco("demo_clip_scene_detection.json")
    [event] = read_scene(FIXTURES / "demo_clip_scene_detection.json").events
    a = native["annotations"][0]
    assert (event.start, event.end, event.frame) == (
        a["start_time"],
        a["end_time"],
        a["frame_start"],
    )
    scene = event.measures[0]
    assert (scene.measure, scene.label, scene.confidence) == (
        "scene",
        "clinic",
        a["score"],
    )
    assert {m.label: m.value for m in event.measures[1:]} == a["all_scores"]
    assert event.box is None  # the whole frame, not a detection


def test_vlm_labels_and_annotated_model():
    native = _coco("demo_clip_vlm_annotation.json")
    contents = read_vlm(FIXTURES / "demo_clip_vlm_annotation.json")
    assert [e.start for e in contents.events] == [
        a["timestamp_sec"] for a in native["annotations"]
    ]
    assert [e.measures[0].label for e in contents.events] == [
        a["label"] for a in native["annotations"]
    ]
    # reasoning equal to the label adds nothing
    assert all(len(e.measures) == 1 for e in contents.events)
    assert contents.annotated_model == "qwen3.5:9b"


def test_rttm_turns():
    events = read_rttm(FIXTURES / "demo_clip_speaker_diarization.rttm").events
    assert [(e.start, e.end, e.track) for e in events] == [
        (0.638, 1.431, "SPEAKER_00"),
        (2.427, 3.44, "SPEAKER_00"),
        (9.312, 9.97, "SPEAKER_00"),
        (9.970, 10.426, "SPEAKER_01"),
    ]
    assert all(e.measures[0].label == e.track for e in events)


def test_webvtt_cues_skip_the_provenance_note():
    events = read_webvtt(FIXTURES / "demo_clip_speech_recognition.vtt").events
    assert (events[0].start, events[0].end, events[0].measures[0].label) == (
        0.0,
        1.32,
        "Ready, baby girl?",
    )
    # The NOTE's JSON escapes "-->", so every arrow left is a cue's.
    text = (FIXTURES / "demo_clip_speech_recognition.vtt").read_text()
    assert len(events) == text.count("-->")
    legacy = read_webvtt(LEGACY / "demo_clip_speech_recognition.vtt").events
    assert [e.measures[0].label for e in legacy] == [
        e.measures[0].label for e in events
    ]


def test_webvtt_multiline_cue_and_hour_less_times(tmp_path):
    path = tmp_path / "x_speech_recognition.vtt"
    path.write_text(
        "WEBVTT\r\n\r\ncue-1\r\n01:02.500 --> 01:04.000\r\nline one\r\nline two\r\n",
        encoding="utf-8",
    )
    [event] = read_webvtt(path).events
    assert (event.start, event.end, event.measures[0].label) == (
        62.5,
        64.0,
        "line one\nline two",
    )


@pytest.mark.parametrize(
    ("reader", "content"),
    [
        (read_person, "{not json"),
        (read_person, '{"images": []}'),
        (read_webvtt, "not a vtt"),
        (read_rttm, "SPEAKER x 1 nope"),
    ],
)
def test_malformed_files_raise_reader_error(tmp_path, reader, content):
    path = tmp_path / "f"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReaderError):
        reader(path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("clip_person_tracking.json", ("person", "clip")),
        ("my_clip_speech_recognition.vtt", ("webvtt", "my_clip")),
        ("clip_person_tracks.json", (None, None)),
        ("clip_openface3_detailed.json", (None, None)),
        ("clip_speaker_diarization.rttm.provenance.json", (None, None)),
        ("clip.MP4", (None, None)),
        ("run.json", (None, None)),
        ("person_tracking.json", (None, None)),
        ("clip.eaf", (None, None)),
        ("notes.txt", (None, "")),
    ],
)
def test_classify(name, expected):
    kind, stem = classify(Path(name))
    assert (kind.kind if kind else None, stem) == expected
