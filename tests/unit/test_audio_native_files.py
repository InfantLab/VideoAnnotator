"""Speech and diarization write the files the viewer loads (spec 014 follow-up).

Only the deprecated `audio_processing` used to write `.vtt` / `.rttm`; running
`speech_recognition` and `speaker_diarization` left none in a job's artifacts,
so the viewer showed no transcript or speaker turns.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from videoannotator.pipelines.audio_processing.native_files import (
    write_rttm,
    write_webvtt,
)

SPEECH_RESULT = {
    "transcript": "Ready, baby girl? Good morning.",
    "metadata": {
        "segments": [
            {"start": 0.0, "end": 1.32, "text": " Ready, baby girl?"},
            {"start": 2.0, "end": 3.16, "text": " Good morning."},
        ]
    },
}


def test_write_webvtt(tmp_path):
    path = tmp_path / "clip_speech_recognition.vtt"
    write_webvtt([SPEECH_RESULT], path)
    text = path.read_text()
    assert text.startswith("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.320\n")
    assert "2\n00:00:02.000 --> 00:00:03.160\n Good morning." in text


def test_write_rttm(tmp_path):
    path = tmp_path / "clip_speaker_diarization.rttm"
    write_rttm(
        [
            {"start_time": 0.64, "duration": 0.79, "speaker_id": "SPEAKER_00"},
            {"start_time": 9.97, "end_time": 10.43, "speaker_id": "SPEAKER_01"},
        ],
        path,
    )
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("SPEAKER ")
    assert "SPEAKER_00" in lines[0] and "SPEAKER_01" in lines[1]


def test_speech_pipeline_writes_vtt_into_output_dir(tmp_path):
    pytest.importorskip("whisper")
    from videoannotator.pipelines.audio_processing.speech_pipeline import (
        SpeechPipeline,
    )

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    pipeline = SpeechPipeline({})
    pipeline.is_initialized = True
    with (
        patch.object(
            SpeechPipeline, "extract_audio_from_video", return_value=(b"", 16000)
        ),
        patch.object(SpeechPipeline, "transcribe_audio", return_value=SPEECH_RESULT),
    ):
        result = pipeline.process(str(video), output_dir=str(tmp_path / "out"))

    assert result == [SPEECH_RESULT]
    vtt = Path(tmp_path / "out" / "clip_speech_recognition.vtt")
    assert vtt.read_text().startswith("WEBVTT")
