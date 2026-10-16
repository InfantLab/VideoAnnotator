"""ELAN files: valid EAF 3.0, one tier per track, matching the table (spec 025, US3)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from videoannotator.exporters.elan import (
    IN_VIEW,
    beside_results,
    build,
    tiers,
    write_videos,
)
from videoannotator.exporters.readers import Row
from videoannotator.exporters.results_set import from_folder
from videoannotator.exporters.table import read_video, video_rows

SCHEMA = Path(__file__).resolve().parents[2] / "fixtures" / "elan" / "EAFv3.0.xsd"


def _parse(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    """Tier id -> [(start ms, end ms, value)], read like the viewer's parser."""
    root = ET.parse(path).getroot()
    slots = {
        s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE")) for s in root.iter("TIME_SLOT")
    }
    out = {}
    for tier in root.iter("TIER"):
        out[tier.get("TIER_ID")] = [
            (
                slots[a.get("TIME_SLOT_REF1")],
                slots[a.get("TIME_SLOT_REF2")],
                a.findtext("ANNOTATION_VALUE") or "",
            )
            for a in tier.iter("ALIGNABLE_ANNOTATION")
        ]
    return out


@pytest.fixture
def eaf(run_folder) -> Path:
    write_videos(from_folder(run_folder), beside_results)
    return run_folder / "demo_clip" / "demo_clip.eaf"


def test_valid_against_the_eaf_schema(eaf):
    xmlschema = pytest.importorskip("xmlschema")
    xmlschema.XMLSchema(str(SCHEMA)).validate(str(eaf))


def test_one_tier_per_pipeline_track(eaf):
    assert list(_parse(eaf)) == [
        "scene_detection scenes",
        "speech_recognition speech",
        "speaker_diarization SPEAKER_00",
        "speaker_diarization SPEAKER_01",
        "person_tracking person_demo_clip_001 (parent)",
        "person_tracking person_demo_clip_002 (parent)",
        "face_analysis faces",
        "face_openface3_embedding faces",
        "vlm_annotation labels",
    ]


def test_intervals_match_the_table(eaf, run_folder):
    video = from_folder(run_folder).videos[0]
    rows = video_rows(video, read_video(video))
    tiers_read = _parse(eaf)
    for measure, tier in (
        ("speech", "speech_recognition speech"),
        ("scene", "scene_detection scenes"),
    ):
        expected = [
            (round(r.start * 1000), round(r.end * 1000), r.label)
            for r in rows
            if r.measure == measure
        ]
        assert tiers_read[tier] == expected
    turns = [
        (round(r.start * 1000), round(r.end * 1000), r.label)
        for r in rows
        if r.measure == "speaker_turn" and r.track == "SPEAKER_00"
    ]
    assert tiers_read["speaker_diarization SPEAKER_00"] == turns


def test_media_is_linked(eaf):
    descriptor = ET.parse(eaf).getroot().find("HEADER/MEDIA_DESCRIPTOR")
    assert descriptor.get("MEDIA_URL") == "file:///data/videos/demo_clip.mp4"
    assert descriptor.get("MIME_TYPE") == "video/mp4"


def _row(start: float, track: str = "p1", measure: str = "person", label="infant"):
    return Row(
        run="",
        video="v",
        pipeline="person_tracking",
        event=1,
        track=track,
        start=start,
        end=None,
        frame=None,
        measure=measure,
        label=label,
        value=None,
        confidence=None,
        x=None,
        y=None,
        width=None,
        height=None,
        model="m",
        videoannotator_version="1",
        file="f",
    )


def test_detections_merge_while_gaps_are_small():
    # Sampled every second; one missed sample (2 s gap) still merges, 3 s splits.
    times = [0, 1, 2, 4, 5, 9, 10]
    [tier], _ = tiers([_row(t) for t in times])
    assert tier.tier_id == "person_tracking p1 (infant)"
    assert [(a.start_ms, a.end_ms, a.value) for a in tier.annotations] == [
        (0, 6000, IN_VIEW),
        (9000, 11000, IN_VIEW),
    ]


def test_spacing_comes_from_the_whole_pipeline():
    # p2 is seen once; its interval lasts one sampling interval of the pipeline.
    rows = [_row(t) for t in (0, 0.5, 1.0)] + [_row(3.0, track="p2")]
    found = {t.tier_id: t.annotations for t in tiers(rows)[0]}
    [single] = found["person_tracking p2 (infant)"]
    assert (single.start_ms, single.end_ms) == (3000, 3500)


def test_a_lone_detection_is_never_zero_length():
    [tier], _ = tiers([_row(2.0)])
    [annotation] = tier.annotations
    assert annotation.end_ms == annotation.start_ms + 1


def test_vlm_labels_hold_until_the_next_sample_and_merge():
    rows = [
        _row(t, track="", measure="vlm_label", label=label)
        for t, label in ((0, "TOUCH"), (2, "TOUCH"), (4, "NO_TOUCH"), (6, "TOUCH"))
    ]
    for r in rows:
        r.pipeline = "vlm_annotation"
    [tier], _ = tiers(rows)
    assert [(a.start_ms, a.end_ms, a.value) for a in tier.annotations] == [
        (0, 4000, "TOUCH"),
        (4000, 6000, "NO_TOUCH"),
        (6000, 8000, "TOUCH"),
    ]


def test_continuous_measures_are_left_out_and_said_so(run_folder):
    summary = write_videos(from_folder(run_folder), beside_results)
    [note] = summary.left_out
    assert "action_unit" in note and "gaze" in note
    assert summary.videos == 1 and summary.rows > 0


def test_legacy_results_export(legacy_folder, tmp_path):
    summary = write_videos(from_folder(legacy_folder), lambda v: tmp_path / "x.eaf")
    assert summary.videos == 1
    root = ET.parse(tmp_path / "x.eaf").getroot()
    assert root.find("HEADER/MEDIA_DESCRIPTOR") is None  # no recorded path
    xmlschema = pytest.importorskip("xmlschema")
    xmlschema.XMLSchema(str(SCHEMA)).validate(str(tmp_path / "x.eaf"))


def test_awkward_text_is_escaped(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "clip_speech_recognition.vtt").write_text(
        'WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<b> & "quotes" 你好\n',
        encoding="utf-8",
    )
    video = from_folder(folder).videos[0]
    tree, count, _ = build(video, video_rows(video, read_video(video)))
    path = tmp_path / "out.eaf"
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    assert _parse(path)["speech_recognition speech"] == [
        (1000, 2000, '<b> & "quotes" 你好')
    ]
    assert count == 1
