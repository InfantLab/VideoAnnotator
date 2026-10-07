"""Stamped outputs still open in the standard readers for their formats (spec 017 FR-007).

Provenance is added as a top-level JSON key, a WebVTT NOTE block and an RTTM
companion file precisely so that these readers see exactly what they saw before.
"""

import json
import shutil
from pathlib import Path

import pytest

from videoannotator.provenance import build_record, stamp_file

FIXTURES = Path(__file__).parents[1] / "fixtures" / "viewer_contract" / "legacy"
RECORD = build_record("test", settings={"note": "a --> b"})


def _stamped(tmp_path: Path, name: str) -> tuple[Path, Path]:
    original = FIXTURES / name
    stamped = tmp_path / name
    shutil.copy(original, stamped)
    stamp_file(stamped, RECORD)
    return original, stamped


@pytest.mark.parametrize(
    "name",
    [
        "demo_clip_person_tracking.json",
        "demo_clip_scene_detection.json",
        "demo_clip_openface3_analysis.json",
        "demo_clip_vlm_annotation.json",
    ],
)
def test_coco_api_loads_stamped_files_identically(tmp_path, name):
    coco_module = pytest.importorskip("pycocotools.coco")
    original, stamped = _stamped(tmp_path, name)
    before, after = coco_module.COCO(str(original)), coco_module.COCO(str(stamped))
    assert after.getAnnIds() == before.getAnnIds()
    assert after.loadAnns(after.getAnnIds()) == before.loadAnns(before.getAnnIds())
    assert json.loads(stamped.read_text())["provenance"]["pipeline"]["name"] == "test"


def test_webvtt_reader_sees_the_same_cues(tmp_path):
    webvtt = pytest.importorskip("webvtt")
    original, stamped = _stamped(tmp_path, "demo_clip_speech_recognition.vtt")
    before, after = webvtt.read(str(original)), webvtt.read(str(stamped))
    assert [(c.start, c.end, c.text) for c in after] == [
        (c.start, c.end, c.text) for c in before
    ]
    assert len(after) > 0


def test_pyannote_loads_the_unchanged_rttm(tmp_path):
    util = pytest.importorskip("pyannote.database.util")
    original, stamped = _stamped(tmp_path, "demo_clip_speaker_diarization.rttm")
    assert stamped.read_bytes() == original.read_bytes()
    before, after = util.load_rttm(str(original)), util.load_rttm(str(stamped))
    assert {k: list(v.itertracks(yield_label=True)) for k, v in after.items()} == {
        k: list(v.itertracks(yield_label=True)) for k, v in before.items()
    }
