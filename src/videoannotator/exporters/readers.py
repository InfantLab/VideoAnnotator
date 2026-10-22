"""Read each pipeline's native output file as events (spec 025).

One reader per format, all producing the same shape: an `Event` (a detection,
a speech segment, a speaker turn, a scene, a VLM label) with one `Measure` for
what it is and one per attribute measured. The table, ELAN and methods exports
all start from here, so they can't disagree about what a file says.

The column-level contract is specs/025-results-out/contracts/table.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TABLE_FORMAT_VERSION = 2
NOT_RECORDED = "not recorded"

COLUMNS = (
    "run",
    "video",
    "pipeline",
    "event",
    "track",
    "start",
    "end",
    "frame",
    "measure",
    "label",
    "value",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "model",
    "videoannotator_version",
    "file",
    # Version 2 (spec 028): who wrote a researcher's note; empty on pipeline rows.
    "author",
)


class ReaderError(Exception):
    """A result file this reader can't make sense of."""


@dataclass
class Measure:
    """One row's worth of an event: what was measured, and the result."""

    measure: str
    label: str | None = None
    value: float | int | None = None
    confidence: float | None = None


@dataclass
class Event:
    """One thing a pipeline found at one time or over one interval."""

    start: float
    measures: list[Measure]
    end: float | None = None
    frame: int | None = None
    track: str = ""
    box: tuple[float, float, float, float] | None = None


@dataclass
class Row:
    """One table row, in COLUMNS order."""

    run: str
    video: str
    pipeline: str
    event: int
    track: str
    start: float
    end: float | None
    frame: int | None
    measure: str
    label: str | None
    value: float | int | None
    confidence: float | None
    x: float | None
    y: float | None
    width: float | None
    height: float | None
    model: str
    videoannotator_version: str
    file: str
    author: str = ""

    def values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, c) for c in COLUMNS)


@dataclass
class FileContents:
    """What a reader found in one file."""

    events: list[Event] = field(default_factory=list)
    # A model named by the file's own annotations (VLM), for files without a
    # provenance record: written by the pipeline, so not a guess.
    annotated_model: str | None = None


Reader = Callable[[Path], FileContents]


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _load_coco(path: Path) -> tuple[list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ReaderError(f"unreadable: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ReaderError("unreadable: no COCO annotations list")
    images = {
        image.get("id"): image
        for image in data.get("images") or []
        if isinstance(image, dict)
    }
    return [a for a in data["annotations"] if isinstance(a, dict)], images


def _time(annotation: dict[str, Any], image: dict[str, Any], *keys: str) -> float:
    for source in (annotation, image):
        for key in keys:
            value = _number(source.get(key))
            if value is not None:
                return value
    raise ReaderError(f"annotation {annotation.get('id')} has no time")


def _frame(annotation: dict[str, Any], image: dict[str, Any]) -> int | None:
    for source in (annotation, image):
        value = source.get("frame_number")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _box(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(_number(v) is not None for v in bbox)
    ):
        return (bbox[0], bbox[1], bbox[2], bbox[3])
    return None


def read_person(path: Path) -> FileContents:
    annotations, images = _load_coco(path)
    events = []
    for a in annotations:
        image = images.get(a.get("image_id"), {})
        track = a.get("person_id") or (
            str(a["track_id"]) if a.get("track_id") is not None else ""
        )
        events.append(
            Event(
                start=_time(a, image, "timestamp"),
                frame=_frame(a, image),
                track=str(track),
                box=_box(a),
                measures=[
                    Measure(
                        "person",
                        label=_text(a.get("person_label")),
                        confidence=_number(a.get("score")),
                    )
                ],
            )
        )
    return FileContents(events)


def _scores(measure: str, scores: Any) -> list[Measure]:
    if not isinstance(scores, dict):
        return []
    return [
        Measure(measure, label=str(name), value=_number(value))
        for name, value in scores.items()
        if _number(value) is not None
    ]


def read_deepface(path: Path) -> FileContents:
    annotations, images = _load_coco(path)
    events = []
    for a in annotations:
        image = images.get(a.get("image_id"), {})
        measures = [Measure("face", confidence=_number(a.get("score")))]
        if a.get("emotion") is not None:
            measures.append(
                Measure(
                    "emotion",
                    label=_text(a["emotion"]),
                    confidence=_number(a.get("emotion_confidence")),
                )
            )
        measures += _scores("emotion_score", a.get("emotion_scores"))
        if _number(a.get("age")) is not None:
            measures.append(Measure("age", value=_number(a["age"])))
        if a.get("gender") is not None:
            measures.append(
                Measure(
                    "gender",
                    label=_text(a["gender"]),
                    confidence=_number(a.get("gender_confidence")),
                )
            )
        # face_id is the face's index within its frame, not a track over time.
        events.append(
            Event(
                start=_time(a, image, "timestamp"),
                frame=_frame(a, image),
                box=_box(a),
                measures=measures,
            )
        )
    return FileContents(events)


def read_openface3(path: Path) -> FileContents:
    annotations, images = _load_coco(path)
    events = []
    for a in annotations:
        image = images.get(a.get("image_id"), {})
        raw = a.get("openface3")
        features: dict[str, Any] = raw if isinstance(raw, dict) else {}
        start = _number(features.get("timestamp"))
        measures = [Measure("face", confidence=_number(features.get("confidence")))]

        units = features.get("action_units")
        if isinstance(units, dict):
            for name, unit in units.items():
                if isinstance(unit, dict):
                    measures.append(
                        Measure("action_unit", name, _number(unit.get("intensity")))
                    )
            for name, unit in units.items():
                if isinstance(unit, dict) and unit.get("presence") is not None:
                    measures.append(
                        Measure(
                            "action_unit_presence",
                            name,
                            _number(bool(unit["presence"])),
                        )
                    )

        for measure, components in (
            ("head_pose", ("pitch", "yaw", "roll")),
            ("gaze", ("direction_x", "direction_y", "direction_z")),
        ):
            part = features.get(measure)
            if isinstance(part, dict):
                confidence = _number(part.get("confidence"))
                measures += [
                    Measure(measure, c, _number(part[c]), confidence)
                    for c in components
                    if _number(part.get(c)) is not None
                ]

        emotion = features.get("emotion")
        if isinstance(emotion, dict):
            probabilities = emotion.get("probabilities")
            dominant = _text(emotion.get("dominant"))
            if dominant is not None:
                confidence = (
                    _number(probabilities.get(dominant))
                    if isinstance(probabilities, dict)
                    else None
                )
                measures.append(Measure("emotion", dominant, confidence=confidence))
            measures += _scores("emotion_score", probabilities)
            for key in ("valence", "arousal"):
                if _number(emotion.get(key)) is not None:
                    measures.append(Measure(key, value=_number(emotion[key])))

        track = features.get("track_id")
        events.append(
            Event(
                start=start if start is not None else _time(a, image, "timestamp"),
                frame=_frame(a, image),
                track=str(track) if track is not None else "",
                box=_box(a),
                measures=measures,
            )
        )
    return FileContents(events)


def read_scene(path: Path) -> FileContents:
    annotations, images = _load_coco(path)
    events = []
    for a in annotations:
        image = images.get(a.get("image_id"), {})
        start = _number(a.get("start_time"))
        frame_start = a.get("frame_start")
        events.append(
            Event(
                start=start if start is not None else _time(a, image, "timestamp"),
                end=_number(a.get("end_time")),
                frame=frame_start if isinstance(frame_start, int) else None,
                measures=[
                    Measure(
                        "scene",
                        label=_text(a.get("scene_type")),
                        confidence=_number(a.get("score")),
                    ),
                    *_scores("scene_score", a.get("all_scores")),
                ],
            )
        )
    return FileContents(events)


def _vlm_field_measures(a: dict[str, Any]) -> list[Measure]:
    """A shaped answer's fields, one measure each (spec 033 R10): numbers in
    `value`, everything else in `label`; then whether it fit the shape."""
    if a.get("fit") is None:
        return []
    out = []
    for name, value in (a.get("fields") or {}).items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            out.append(Measure(f"vlm.{name}", value=value))
        else:
            out.append(Measure(f"vlm.{name}", label=_text(value)))
    out.append(Measure("vlm_fit", label="yes" if a["fit"] else "no"))
    if not a["fit"] and a.get("fit_problem"):
        out.append(Measure("vlm_fit_problem", label=_text(a["fit_problem"])))
    return out


def read_vlm(path: Path) -> FileContents:
    annotations, images = _load_coco(path)
    events = []
    models = []
    for a in annotations:
        image = images.get(a.get("image_id"), {})
        label = _text(a.get("label"))
        measures = [Measure("vlm_label", label=label)]
        reasoning = _text(a.get("reasoning"))
        if reasoning is not None and reasoning != label:
            measures.append(Measure("vlm_reasoning", label=reasoning))
        measures += _vlm_field_measures(a)
        events.append(
            Event(
                start=_time(a, image, "timestamp_sec", "timestamp"),
                frame=_frame(a, image),
                measures=measures,
            )
        )
        if a.get("model") and a["model"] not in models:
            models.append(str(a["model"]))
    return FileContents(events, " + ".join(models) or None)


def read_rttm(path: Path) -> FileContents:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise ReaderError(f"unreadable: {e}") from e
    events = []
    for number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        try:
            start, duration, speaker = float(fields[3]), float(fields[4]), fields[7]
        except (IndexError, ValueError) as e:
            raise ReaderError(f"unreadable: line {number} is not RTTM") from e
        events.append(
            Event(
                start=start,
                # RTTM gives a duration; keep the sum to RTTM's own precision.
                end=round(start + duration, 6),
                track=speaker,
                measures=[Measure("speaker_turn", label=speaker)],
            )
        )
    return FileContents(events)


_CUE_TIME = re.compile(
    r"^\s*((?:\d+:)?\d{1,2}:\d{2}\.\d{3})\s+-->\s+((?:\d+:)?\d{1,2}:\d{2}\.\d{3})"
)


def _seconds(stamp: str) -> float:
    parts = [float(p) for p in stamp.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return round(seconds, 3)


def read_webvtt(path: Path) -> FileContents:
    try:
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError) as e:
        raise ReaderError(f"unreadable: {e}") from e
    if not text.startswith("WEBVTT"):
        raise ReaderError("unreadable: not a WebVTT file")
    events = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.split("\n")
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        for i, line in enumerate(lines[:2]):
            match = _CUE_TIME.match(line)
            if match:
                cue = "\n".join(lines[i + 1 :]).strip()
                events.append(
                    Event(
                        start=_seconds(match[1]),
                        end=_seconds(match[2]),
                        measures=[Measure("speech", label=cue or None)],
                    )
                )
                break
    return FileContents(events)


@dataclass(frozen=True)
class FileKind:
    """A native output file this module reads."""

    suffix: str
    kind: str
    default_pipeline: str
    reader: Reader


FILE_KINDS = (
    FileKind("person_tracking.json", "person", "person_tracking", read_person),
    FileKind("face_detections.json", "face_deepface", "face_analysis", read_deepface),
    FileKind(
        "openface3_analysis.json",
        "face_openface3",
        "face_openface3_embedding",
        read_openface3,
    ),
    FileKind("scene_detection.json", "scene", "scene_detection", read_scene),
    FileKind("vlm_annotation.json", "vlm", "vlm_annotation", read_vlm),
    FileKind("speaker_diarization.rttm", "rttm", "speaker_diarization", read_rttm),
    FileKind("speech_recognition.vtt", "webvtt", "speech_recognition", read_webvtt),
)

# Restate what another file says (labels on each detection; OpenFace's
# analysis file), so reading them would count facts twice.
RESTATING_SUFFIXES = ("person_tracks.json", "openface3_detailed.json")

PIPELINE_NAMES = {k.default_pipeline for k in FILE_KINDS} | {"audio_processing"}
EXPORT_SUFFIXES = (".csv", ".parquet", ".eaf", ".bib")
IGNORED_NAMES = {"run.json", "job_metadata.json", "methods.md"}


def classify(path: Path) -> tuple[FileKind | None, str | None]:
    """`(kind, video stem)` for a result file; `(None, None)` for one that
    isn't a result (a video, run.json, an export, a provenance companion);
    `(None, "")` for a file no reader knows."""
    from ..provenance import COMPANION_SUFFIX
    from ..results_folder import VIDEO_SUFFIXES

    name = path.name
    lower = name.lower()
    if (
        name.startswith(".")
        or lower.endswith(COMPANION_SUFFIX)
        or lower.endswith(VIDEO_SUFFIXES)
        or lower.endswith(EXPORT_SUFFIXES)
        or name in IGNORED_NAMES
        or (path.suffix == ".json" and path.stem in PIPELINE_NAMES)
        or lower.endswith(tuple("_" + s for s in RESTATING_SUFFIXES))
    ):
        return None, None
    for kind in FILE_KINDS:
        if name.endswith("_" + kind.suffix):
            return kind, name[: -len(kind.suffix) - 1]
    return None, ""
