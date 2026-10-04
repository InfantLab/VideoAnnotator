"""Provenance records and how they are stamped into output files (spec 017)."""

import json

from videoannotator import provenance
from videoannotator.provenance import (
    ModelRef,
    build_record,
    companion_path,
    file_sha256,
    read_record,
    redact,
    stamp_file,
    weights_ref,
)
from videoannotator.version import __version__


def _record(**kwargs):
    return build_record("person_tracking", **kwargs)


def test_record_has_every_field():
    record = _record(
        models=[ModelRef("yolo11n-pose.pt", "file", "ab12", "sha256")],
        settings={"conf_threshold": 0.4},
        determinism={"deterministic": True},
        job_id="job-1",
        input_name="clip.mp4",
        input_sha256="ff00",
    )
    assert record["schema_version"] == 1
    assert record["pipeline"] == {"name": "person_tracking", "sub_pipeline": None}
    assert record["videoannotator_version"] == __version__
    assert record["models"] == [
        {"name": "yolo11n-pose.pt", "source": "file", "revision": "ab12",
         "revision_kind": "sha256"}
    ]  # fmt: skip
    assert record["created_at"].endswith("+00:00")
    assert record["input"] == {"name": "clip.mp4", "sha256": "ff00"}
    assert "vlm" not in record


def test_secrets_are_redacted_at_any_depth():
    settings = {
        "huggingface_token": "hf_x",
        "nested": {"api_key": "k", "auth": "a", "author": "Ada"},
        "list": [{"password": "p"}],
        "empty_token": "",
        "use_auth_token": True,
        "threshold": 0.5,
    }
    assert redact(settings) == {
        "huggingface_token": "<redacted>",
        "nested": {"api_key": "<redacted>", "auth": "<redacted>", "author": "Ada"},
        "list": [{"password": "<redacted>"}],
        "empty_token": "",
        "use_auth_token": True,
        "threshold": 0.5,
    }


def test_weights_hash_is_cached(tmp_path, monkeypatch):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    first = file_sha256(weights)
    calls = []
    real_sha = provenance.hashlib.sha256
    monkeypatch.setattr(
        provenance.hashlib, "sha256", lambda: calls.append(1) or real_sha()
    )
    assert file_sha256(weights) == first
    assert calls == []
    assert weights_ref("w", "file", tmp_path / "missing.pt").revision_note


def test_json_round_trip_and_restamp(tmp_path):
    path = tmp_path / "clip_person_tracking.json"
    path.write_text(json.dumps({"images": [], "annotations": [{"id": 1}]}))
    stamp_file(path, _record(job_id="a"))
    stamp_file(path, _record(job_id="b"))
    data = json.loads(path.read_text())
    assert data["annotations"] == [{"id": 1}]
    assert read_record(path)["job_id"] == "b"


def test_webvtt_note_round_trip(tmp_path):
    path = tmp_path / "clip_speech_recognition.vtt"
    path.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nHi\n\n")
    record = _record(settings={"prompt": "a --> b"})
    stamp_file(path, record)
    stamp_file(path, record)
    text = path.read_text()
    assert text.count(provenance.VTT_MARKER) == 1
    note = next(b for b in text.split("\n\n") if b.startswith("NOTE"))
    assert "-->" not in note and "\n" not in note
    assert text.startswith("WEBVTT\n\nNOTE ")
    assert "1\n00:00:00.000 --> 00:00:01.000\nHi" in text
    assert read_record(path)["settings"]["prompt"] == "a --> b"


def test_rttm_gets_a_companion(tmp_path):
    path = tmp_path / "clip_speaker_diarization.rttm"
    original = "SPEAKER clip 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    path.write_text(original)
    stamp_file(path, _record())
    assert path.read_text() == original
    assert companion_path(path).name == "clip_speaker_diarization.rttm.provenance.json"
    assert read_record(path)["pipeline"]["name"] == "person_tracking"


def test_unstamped_file_has_no_record(tmp_path):
    path = tmp_path / "old.json"
    path.write_text("{}")
    assert read_record(path) is None
