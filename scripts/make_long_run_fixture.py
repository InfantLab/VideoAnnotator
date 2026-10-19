#!/usr/bin/env python3
"""Write a synthetic hour of results for the Video page's performance tests.

Spec 028 SC-004: zooming, panning and playing stay smooth with an hour-long
video and 50,000 events. Real outputs that long are too big to commit, so this
writes deterministic stand-ins (seeded) in the pipelines' own formats:

- `long_run_person_tracking.json`: COCO, 4 tracks sampled at 5 fps, each seen
  in stretches with gaps (about 40,000 detections over 3,600 s);
- `long_run_speech_recognition.vtt`: 2,000 cues;
- `long_run_speaker_diarization.rttm`: 3 speakers, 6,000 turns.

Standard library only, so the viewer's tests can run it wherever Python 3 is.

    python3 scripts/make_long_run_fixture.py [out_dir]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DURATION = 3600.0
FPS = 5
STEM = "long_run"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "video_page"
    / "long_run"
)


def _clock(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def person_tracking(rng: random.Random) -> dict:
    frames = int(DURATION * FPS)
    images = [
        {
            "id": f"{STEM}_frame_{i * 6}",
            "width": 1280,
            "height": 720,
            "file_name": f"frame_{i * 6:06d}.jpg",
            "video_id": STEM,
            "frame_number": i * 6,
            "timestamp": round(i / FPS, 6),
        }
        for i in range(frames)
    ]
    annotations = []
    for track in range(1, 5):
        i = rng.randrange(0, 50)
        while i < frames:
            seen = rng.randint(25, 400)
            x, y = rng.uniform(0, 900), rng.uniform(0, 300)
            for j in range(i, min(frames, i + seen)):
                x = min(1000.0, max(0.0, x + rng.uniform(-4, 4)))
                annotations.append(
                    {
                        "id": 0,
                        "image_id": images[j]["id"],
                        "category_id": 1,
                        "bbox": [round(x, 2), round(y, 2), 220.0, 400.0],
                        "area": 88000.0,
                        "iscrowd": 0,
                        "score": round(rng.uniform(0.5, 0.99), 4),
                        "keypoints": [],
                        "num_keypoints": 0,
                        "track_id": track,
                        "timestamp": images[j]["timestamp"],
                        "frame_number": images[j]["frame_number"],
                        "person_id": f"person_{STEM}_{track:03d}",
                        "person_label": "parent" if track % 2 else "infant",
                    }
                )
            i += seen + rng.randint(25, 300)
    annotations.sort(key=lambda a: (a["timestamp"], a["track_id"]))
    for number, a in enumerate(annotations, 1):
        a["id"] = number
    return {
        "info": {"description": "VideoAnnotator COCO Export (synthetic)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }


def webvtt(rng: random.Random) -> str:
    words = [
        "peekaboo",
        "where",
        "is",
        "baby",
        "there",
        "you",
        "are",
        "ready",
        "look",
        "at",
        "me",
    ]
    lines = ["WEBVTT", ""]
    step = DURATION / 2000
    for n in range(2000):
        start = n * step + rng.uniform(0, step * 0.3)
        end = start + rng.uniform(0.3, step * 0.65)
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
        lines += [f"{_clock(start)} --> {_clock(end)}", text, ""]
    return "\n".join(lines)


def rttm(rng: random.Random) -> str:
    lines = []
    step = DURATION / 6000
    for n in range(6000):
        start = n * step + rng.uniform(0, step * 0.2)
        length = rng.uniform(0.1, step * 0.75)
        speaker = f"SPEAKER_{rng.randrange(3):02d}"
        lines.append(
            f"SPEAKER {STEM} 1 {start:.3f} {length:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines) + "\n"


def main(out: Path) -> None:
    rng = random.Random(28)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{STEM}_person_tracking.json").write_text(
        json.dumps(person_tracking(rng), separators=(",", ":")), encoding="utf-8"
    )
    (out / f"{STEM}_speech_recognition.vtt").write_text(webvtt(rng), encoding="utf-8")
    (out / f"{STEM}_speaker_diarization.rttm").write_text(rttm(rng), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
