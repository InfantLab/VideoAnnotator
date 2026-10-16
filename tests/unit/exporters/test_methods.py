"""The methods paragraph says exactly what the provenance records (spec 025, US2)."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from videoannotator.exporters.methods import describe
from videoannotator.exporters.results_set import from_folder

from .conftest import FIXTURES, LEGACY, make_run

CITED = re.compile(r"@([\w-]+)")
ENTRY = re.compile(r"^@\w+\{([^,\s]+),", re.MULTILINE)


def _records() -> list[dict]:
    records = []
    for path in sorted(FIXTURES.glob("demo_clip_*")):
        if path.suffix == ".json" and not path.name.endswith(".provenance.json"):
            record = json.loads(path.read_text()).get("provenance")
        elif path.name.endswith(".provenance.json"):
            record = json.loads(path.read_text())
        elif path.suffix == ".vtt":
            line = next(
                text
                for text in path.read_text().splitlines()
                if text.startswith("NOTE videoannotator")
            )
            record = json.loads(line.split(" ", 2)[2])
        else:
            continue
        if record:
            records.append(record)
    return records


def test_every_recorded_revision_is_in_the_text_exactly(run_folder):
    """SC-005."""
    described = describe(from_folder(run_folder))
    for record in _records():
        for model in record["models"]:
            assert model["revision"][:12] in described.prose
            assert any(
                r.endswith(f"{model['revision_kind']} {model['revision']}")
                for r in described.revisions
            ), model


def test_every_cited_key_is_in_the_bibtex(run_folder):
    described = describe(from_folder(run_folder))
    cited = set(CITED.findall(described.prose))
    entries = ENTRY.findall(described.bibtex)
    assert cited and cited <= set(entries)
    assert entries[0] == "addyman2026videoannotator"
    assert len(entries) == len(set(entries))


def test_one_group_is_described_once(run_folder):
    described = describe(from_folder(run_folder))
    assert described.prose.startswith(
        "The video was annotated automatically with VideoAnnotator 1.5.0"
    )
    assert described.summary["groups"] == 1
    assert "Speech was transcribed with whisper base" in described.prose
    assert "GPU deterministic mode was off." in described.prose


def test_two_identical_videos_are_one_group(tmp_path):
    run = make_run(tmp_path, {"a": FIXTURES, "b": FIXTURES})
    described = describe(from_folder(run))
    assert described.prose.startswith("All 2 videos were annotated")
    assert described.summary == {
        "videos": 2,
        "groups": 1,
        "not_recorded": ["face_analysis", "vlm_annotation"],
    }


def test_a_changed_setting_is_stated_with_its_count(tmp_path):
    run = make_run(tmp_path, {"a": FIXTURES, "b": FIXTURES, "c": FIXTURES})
    rttm_record = run / "c" / "demo_clip_speaker_diarization.rttm.provenance.json"
    record = json.loads(rttm_record.read_text())
    record["settings"]["max_speakers"] = 2
    rttm_record.write_text(json.dumps(record))
    described = describe(from_folder(run))
    assert described.prose.startswith("Of 3 videos, 2 were annotated")
    assert "For the other video, speaker turns were identified with" in described.prose
    assert "max_speakers = 2" in described.prose


def test_results_without_provenance_say_not_recorded(legacy_folder):
    described = describe(from_folder(legacy_folder))
    assert "VideoAnnotator (version not recorded)" in described.prose
    assert "Speech was transcribed (model version not recorded)" in described.prose
    assert described.revisions == []
    # Nothing filled in from today's defaults.
    assert "whisper base" not in described.prose


def test_vlm_prompt_is_identified_and_listed(tmp_path):
    run = make_run(tmp_path, {"a": FIXTURES})
    path = run / "a" / "demo_clip_vlm_annotation.json"
    data = json.loads(path.read_text())
    prompt = data["annotations"][0]["prompt"]
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    data["provenance"] = {
        "schema_version": 1,
        "pipeline": {"name": "vlm_annotation", "sub_pipeline": None},
        "videoannotator_version": "1.5.0",
        "models": [
            {
                "name": "qwen3.5:9b",
                "source": "ollama",
                "revision": "d" * 64,
                "revision_kind": "ollama-digest",
            }
        ],
        "settings": {
            "prompt": prompt,
            "model": "qwen3.5:9b",
            "base_url": "http://lab-server:11434",
        },
        "determinism": {},
        "vlm": {
            "prompt_sha256": sha,
            "model_digest": "d" * 64,
            "quantization": "Q4_K_M",
            "base_url": "http://lab-server:11434",
        },
    }
    path.write_text(json.dumps(data))
    described = describe(from_folder(run))
    assert f"qwen3.5:9b (digest {'d' * 12})" in described.prose
    assert f"using prompt {sha[:12]}" in described.prose
    assert "quantised Q4_K_M" in described.prose
    assert described.prompts == [{"sha256": sha, "text": prompt}]
    assert "lab-server" not in described.prose
    assert prompt in described.to_markdown()


def test_no_names_paths_or_ids_in_the_prose(tmp_path):
    """FR-029."""
    run = make_run(tmp_path, {"P017_mother_infant": FIXTURES})
    described = describe(from_folder(run))
    job_id = _records()[0]["job_id"]
    for leak in (
        "P017",
        "demo_clip",
        str(tmp_path),
        "/data/videos",
        job_id,
        "/app/models",
    ):
        assert leak not in described.prose


def test_markdown_has_prose_revisions_and_bibtex(run_folder):
    markdown = describe(from_folder(run_folder)).to_markdown()
    assert markdown.startswith("# Methods")
    assert "## Model revisions" in markdown
    assert "```bibtex\n@software{addyman2026videoannotator," in markdown


def test_nothing_to_describe(tmp_path):
    with pytest.raises(ValueError):
        describe(from_folder(tmp_path))


def test_legacy_and_current_differ_by_version(tmp_path):
    run = make_run(tmp_path, {"new": FIXTURES, "old": LEGACY})
    described = describe(from_folder(run))
    assert "VideoAnnotator (version not recorded) was used" in described.prose


def test_local_weights_are_named_by_file_not_path(tmp_path):
    run = make_run(tmp_path, {"a": FIXTURES})
    path = run / "a" / "demo_clip_person_tracking.json"
    data = json.loads(path.read_text())
    data["provenance"]["models"][0].update(revision=None, revision_kind="unknown")
    path.write_text(json.dumps(data))
    prose = describe(from_folder(run)).prose
    assert "yolo11n-pose.pt (revision not recorded)" in prose
    assert "models/yolo" not in prose
    assert "pyannote/speaker-diarization-3.1" in prose  # a hub id keeps its owner
