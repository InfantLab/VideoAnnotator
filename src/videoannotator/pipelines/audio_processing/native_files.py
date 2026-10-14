"""Write speech and diarization results as the files the viewer loads.

Shared by `speech_recognition`, `speaker_diarization` and the deprecated
`audio_processing`, which used to be the only one that wrote them: running the
two standalone pipelines left no `.vtt` / `.rttm` in the job's artifacts, so the
viewer showed no transcript or speaker turns (found 2026-10-01, spec 014).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...exporters.native_formats import export_rttm_diarization


def _vtt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_webvtt(speech_results: list[dict[str, Any]], path: Path) -> None:
    """Write Whisper segments (from the first result's metadata) as WebVTT."""
    speech = speech_results[0] if speech_results else {}
    segments = speech.get("metadata", {}).get("segments", [])
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, start=1):
            start, end = _vtt_time(seg["start"]), _vtt_time(seg["end"])
            f.write(f"{idx}\n{start} --> {end}\n{seg['text']}\n\n")


def write_rttm(turns: list[dict[str, Any]], path: Path) -> None:
    """Write diarization turns (start_time/duration/speaker_id) as RTTM."""
    segments = [
        {
            "start": turn.get("start_time", 0.0),
            "end": turn.get(
                "end_time", turn.get("start_time", 0.0) + turn.get("duration", 0.0)
            ),
            "speaker_id": turn.get("speaker_id"),
        }
        for turn in turns
    ]
    export_rttm_diarization(segments, str(path))
