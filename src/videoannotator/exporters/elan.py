"""ELAN (.eaf) files from the results table (spec 025, US3).

One file per video, one tier per pipeline track, built from the table's rows,
so ELAN and the table can't disagree. Intervals (speech, speaker turns,
scenes) go in as they are. Frame-by-frame detections become "in view"
annotations: consecutive detections of one track merge while the gap is at
most twice the pipeline's sampling interval, and each run lasts until one
interval after its last detection. VLM labels hold until the next sample, as
the viewer draws them (constitution Principle VI). Continuous measures (action
units, gaze, scores) and per-detection attributes (emotion, age) stay in the
table: ELAN codes what happens when, not every frame's measurements.

Contract: specs/025-results-out/contracts/elan-and-methods.md.
"""

from __future__ import annotations

import mimetypes
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..version import __version__
from .readers import TABLE_FORMAT_VERSION, Row
from .results_set import ResultsSet, VideoResults
from .table import ExportSummary, read_video, video_rows

LINGUISTIC_TYPE = "videoannotator"
IN_VIEW = "in view"
SCHEMA_URL = "http://www.mpi.nl/tools/elan/EAFv3.0.xsd"

_INTERVALS = {"scene": "scenes", "speech": "speech"}
_DETECTIONS = {"person", "face"}
_TIER_ORDER = ("scene", "speech", "speaker_turn", "person", "face", "vlm_label")


@dataclass
class Annotation:
    start_ms: int
    end_ms: int
    value: str


@dataclass
class Tier:
    tier_id: str
    order: tuple[int, str]
    annotations: list[Annotation]


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _annotation(start: float, end: float, value: str) -> Annotation:
    start_ms, end_ms = _ms(start), _ms(end)
    return Annotation(start_ms, max(end_ms, start_ms + 1), value)


def _spacing(times: list[float]) -> float:
    gaps = [b - a for a, b in pairwise(times) if b > a]
    return statistics.median(gaps) if gaps else 0.0


def _in_view(times: list[float], spacing: float) -> list[Annotation]:
    """Runs of detections at `times` (sorted, unique) as "in view" intervals."""
    runs: list[list[float]] = []
    for t in times:
        if runs and t - runs[-1][-1] <= 2 * spacing:
            runs[-1].append(t)
        else:
            runs.append([t])
    return [_annotation(run[0], run[-1] + spacing, IN_VIEW) for run in runs]


def _held(samples: list[tuple[float, str]], spacing: float) -> list[Annotation]:
    """Labels at sample times, each lasting until the next sample."""
    out: list[Annotation] = []
    for i, (start, label) in enumerate(samples):
        end = samples[i + 1][0] if i + 1 < len(samples) else start + spacing
        if out and out[-1].value == label and out[-1].end_ms == _ms(start):
            out[-1].end_ms = max(_ms(end), out[-1].start_ms + 1)
        else:
            out.append(_annotation(start, end, label))
    return out


def tiers(rows: list[Row]) -> tuple[list[Tier], list[str]]:
    """The tiers for one video's rows, and the measures left out."""
    result: list[Tier] = []
    left_out: set[str] = set()
    pipelines_times: dict[str, set[float]] = defaultdict(set)
    detections: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    vlm: dict[str, list[Row]] = defaultdict(list)
    intervals: dict[tuple[str, str, str], list[Row]] = defaultdict(list)

    for row in rows:
        if row.measure in _DETECTIONS:
            detections[(row.pipeline, row.measure, row.track)].append(row)
            pipelines_times[row.pipeline].add(row.start)
        elif row.measure == "vlm_label":
            vlm[row.pipeline].append(row)
        elif row.measure in _INTERVALS or row.measure == "speaker_turn":
            if row.end is not None:
                intervals[(row.pipeline, row.measure, row.track)].append(row)
        elif row.measure != "vlm_reasoning":
            left_out.add(row.measure)

    for (pipeline, measure, track), group in intervals.items():
        name = _INTERVALS.get(measure) or track or "speakers"
        result.append(
            Tier(
                f"{pipeline} {name}",
                (_TIER_ORDER.index(measure), f"{pipeline} {name}"),
                [_annotation(r.start, r.end or r.start, r.label or "") for r in group],
            )
        )

    for (pipeline, measure, track), group in detections.items():
        spacing = _spacing(sorted(pipelines_times[pipeline]))
        if track:
            labels = Counter(r.label for r in group if r.label)
            name = f"{track} ({labels.most_common(1)[0][0]})" if labels else track
        else:
            name = "faces" if measure == "face" else "people"
        times = sorted({r.start for r in group})
        result.append(
            Tier(
                f"{pipeline} {name}",
                (_TIER_ORDER.index(measure), f"{pipeline} {name}"),
                _in_view(times, spacing),
            )
        )

    for pipeline, group in vlm.items():
        samples = sorted((r.start, r.label or "") for r in group)
        spacing = _spacing([s for s, _ in samples])
        result.append(
            Tier(
                f"{pipeline} labels",
                (_TIER_ORDER.index("vlm_label"), pipeline),
                _held(samples, spacing),
            )
        )

    result.sort(key=lambda t: t.order)
    used: Counter[str] = Counter()
    for tier in result:
        used[tier.tier_id] += 1
        if used[tier.tier_id] > 1:
            tier.tier_id = f"{tier.tier_id} {used[tier.tier_id]}"
    return result, sorted(left_out)


def _media_url(path: str) -> str | None:
    for flavour in (PurePosixPath, PureWindowsPath):
        candidate = flavour(path)
        if candidate.is_absolute():
            return candidate.as_uri()
    return None


def build(
    video: VideoResults, rows: list[Row]
) -> tuple[ET.ElementTree, int, list[str]]:
    """The .eaf document for `video`, its annotation count, and measures left out."""
    video_tiers, left_out = tiers(rows)
    root = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": f"VideoAnnotator {__version__}",
            "DATE": datetime.now(UTC).isoformat(timespec="seconds"),
            "FORMAT": "3.0",
            "VERSION": "3.0",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": SCHEMA_URL,
        },
    )
    header = ET.SubElement(
        root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"}
    )
    url = _media_url(video.media_path) if video.media_path else None
    if url is not None:
        ET.SubElement(
            header,
            "MEDIA_DESCRIPTOR",
            {
                "MEDIA_URL": url,
                "MIME_TYPE": mimetypes.guess_type(url)[0] or "video/*",
            },
        )
    total = sum(len(t.annotations) for t in video_tiers)
    for name, value in (
        ("lastUsedAnnotationId", str(total)),
        ("videoannotator.table_format_version", str(TABLE_FORMAT_VERSION)),
    ):
        ET.SubElement(header, "PROPERTY", {"NAME": name}).text = value

    time_order = ET.SubElement(root, "TIME_ORDER")
    slots: list[tuple[int, str]] = []
    annotation_number = 0
    tier_elements = []
    for tier in video_tiers:
        element = ET.Element(
            "TIER", {"LINGUISTIC_TYPE_REF": LINGUISTIC_TYPE, "TIER_ID": tier.tier_id}
        )
        for a in tier.annotations:
            annotation_number += 1
            first, second = f"ts{len(slots) + 1}", f"ts{len(slots) + 2}"
            slots += [(a.start_ms, first), (a.end_ms, second)]
            alignable = ET.SubElement(
                ET.SubElement(element, "ANNOTATION"),
                "ALIGNABLE_ANNOTATION",
                {
                    "ANNOTATION_ID": f"a{annotation_number}",
                    "TIME_SLOT_REF1": first,
                    "TIME_SLOT_REF2": second,
                },
            )
            ET.SubElement(alignable, "ANNOTATION_VALUE").text = a.value
        tier_elements.append(element)
    for ms, slot_id in sorted(slots, key=lambda s: (s[0], int(s[1][2:]))):
        ET.SubElement(
            time_order, "TIME_SLOT", {"TIME_SLOT_ID": slot_id, "TIME_VALUE": str(ms)}
        )
    root.extend(tier_elements)
    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": LINGUISTIC_TYPE,
            "TIME_ALIGNABLE": "true",
        },
    )
    tree = ET.ElementTree(root)
    ET.indent(tree)
    return tree, total, left_out


def write_tree(tree: ET.ElementTree, path: Path) -> None:
    """Write an .eaf as UTF-8 with a declaration, ending in a newline."""
    root = tree.getroot()
    if root is None:
        raise ValueError("an .eaf needs a root element")
    path.write_bytes(ET.tostring(root, encoding="UTF-8", xml_declaration=True) + b"\n")


def _left_out_note(measures: set[str]) -> list[str]:
    if not measures:
        return []
    return [
        "Not in ELAN, but in the table: per-detection measures ("
        + ", ".join(sorted(measures))
        + ")"
    ]


def write_videos(
    results: ResultsSet, target: Callable[[VideoResults], Path]
) -> ExportSummary:
    """Write one .eaf per video to `target(video) -> Path`."""
    summary = ExportSummary()
    left_out: set[str] = set()
    pipelines: set[str] = set()
    for video in results.videos:
        rows = video_rows(video, read_video(video))
        if not rows:
            continue
        tree, count, measures = build(video, rows)
        path = target(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_tree(tree, path)
        summary.rows += count
        summary.videos += 1
        left_out.update(measures)
        pipelines.update(r.pipeline for r in rows)
    summary.pipelines = sorted(pipelines)
    summary.skipped = [s.to_dict() for s in results.skipped]
    summary.missing = [m.to_dict() for m in results.missing]
    summary.notes = list(results.notes)
    summary.left_out = _left_out_note(left_out)
    return summary


def file_name(video: VideoResults) -> str:
    return f"{video.video}.eaf"


def beside_results(video: VideoResults) -> Path:
    """The default place for a video's .eaf: the folder holding its results."""
    return video.files[0].path.parent / file_name(video)
