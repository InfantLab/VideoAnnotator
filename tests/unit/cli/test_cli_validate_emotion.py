import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()

VALID = {
    "schema_version": 1,
    "source_pipeline": "face_laion_clip",
    "emotions": [
        {
            "start": 0.0,
            "end": 0.4,
            "labels": [
                {"label": "happy", "confidence": 0.9},
                {"label": "neutral", "confidence": 0.05},
            ],
            "confidence": 0.9,
            "source": {"modality": "video"},
        }
    ],
}

INVALID = {
    "schema_version": 1,
    "source_pipeline": "face_laion_clip",
    "emotions": [
        {
            "start": 0.4,
            "end": 0.1,  # invalid ordering
            "labels": [{"label": "happy", "confidence": 1.1}],
        }
    ],
}


def test_validate_emotion_ok(tmp_path: Path):
    p = tmp_path / "valid.emotion.json"
    p.write_text(json.dumps(VALID), encoding="utf-8")
    result = runner.invoke(app, ["validate-emotion", str(p)])
    assert result.exit_code == 0
    assert "[OK]" in result.stdout


def test_validate_emotion_fail(tmp_path: Path):
    p = tmp_path / "invalid.emotion.json"
    p.write_text(json.dumps(INVALID), encoding="utf-8")
    result = runner.invoke(app, ["validate-emotion", str(p)])
    assert result.exit_code != 0
    assert "[ERROR]" in result.stdout
    assert "end must be greater" in result.stdout or "out of range" in result.stdout
