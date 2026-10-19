"""One table of every pipeline's results, as CSV or Parquet (spec 025).

Rows come one video at a time, so a large run never has to fit in memory:
the sort key starts with the video. Contract:
specs/025-results-out/contracts/table.md.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .readers import (
    COLUMNS,
    NOT_RECORDED,
    TABLE_FORMAT_VERSION,
    Event,
    FileContents,
    ReaderError,
    Row,
)
from .results_set import ResultFile, ResultsSet, SkippedFile, VideoResults

PARQUET_EXTRA = "parquet"
NOTES_PIPELINE = "notes"
NOTE_MEASURE = "note"
NO_NOTES_DATABASE = "notes not included: no database"


class NotesUnavailable(RuntimeError):
    """The notes database can't be read (an export of a copied folder)."""


# A video's researcher notes (spec 028): objects with `start_sec`, `end_sec`,
# `text`, `author_name` and `video_path`, in table order.
NotesFor = Callable[[VideoResults], list[Any]]
PARQUET_HINT = f"pip install videoannotator[{PARQUET_EXTRA}]"


class ParquetNotInstalled(RuntimeError):
    """Parquet was asked for, but its optional component isn't installed."""

    def __init__(self) -> None:
        super().__init__(
            "Parquet export needs the optional 'parquet' component. "
            f"Install it with: {PARQUET_HINT}"
        )


@dataclass
class ExportSummary:
    """What an export covered, and what it left out and why (FR-013)."""

    rows: int = 0
    videos: int = 0
    pipelines: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    missing: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    # Researchers' notes included as `note` rows (table format 2).
    researcher_notes: int = 0

    def to_dict(self, *, sources: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "table_format_version": TABLE_FORMAT_VERSION,
            "rows": self.rows,
            "videos": self.videos,
            "pipelines": self.pipelines,
            "skipped": self.skipped,
            "missing": self.missing,
            "notes": self.notes,
            "left_out": self.left_out,
            "researcher_notes": self.researcher_notes,
            "truncated": False,
        }
        if sources:
            data["sources"] = self.sources
        return data

    def to_header(self, max_bytes: int = 8192) -> str:
        """ASCII JSON of at most `max_bytes`; long lists are cut (`truncated`)."""
        data = self.to_dict()
        text = json.dumps(data, separators=(",", ":"))
        while len(text) > max_bytes:
            longest = max(("skipped", "missing", "notes"), key=lambda k: len(data[k]))
            if not data[longest]:
                break
            data[longest] = data[longest][: len(data[longest]) // 2]
            data["truncated"] = True
            text = json.dumps(data, separators=(",", ":"))
        return text

    def lines(self) -> list[str]:
        """The summary for people: skipped files, videos without results, notes."""
        out = [
            f"skipped {s['file']} ({s['video']}): {s['reason']}" for s in self.skipped
        ]
        out += [f"no results for {m['video']}: {m['reason']}" for m in self.missing]
        return out + self.notes + self.left_out


def _model(result: ResultFile, contents: FileContents) -> str:
    models = (result.provenance or {}).get("models")
    if isinstance(models, list) and models:
        names = [
            str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name")
        ]
        if names:
            return " + ".join(names)
    if result.provenance is None and contents.annotated_model:
        return contents.annotated_model
    return NOT_RECORDED


def _version(result: ResultFile) -> str:
    version = (result.provenance or {}).get("videoannotator_version")
    return str(version) if version else NOT_RECORDED


@dataclass
class _ReadFile:
    result: ResultFile
    contents: FileContents


def read_video(video: VideoResults) -> list[_ReadFile]:
    """Every readable file of `video`; unreadable ones join `video.skipped`."""
    read = []
    for result in video.files:
        try:
            read.append(_ReadFile(result, result.kind.reader(result.path)))
        except ReaderError as e:
            video.skipped.append(
                SkippedFile(video.run, video.video, result.path.name, str(e))
            )
    return read


def video_rows(video: VideoResults, read: list[_ReadFile] | None = None) -> list[Row]:
    """`video`'s rows in table order (FR-015)."""
    numbered: list[tuple[str, int, Event, _ReadFile]] = []
    counters: dict[str, int] = {}
    for item in read if read is not None else read_video(video):
        pipeline = item.result.pipeline
        for event in item.contents.events:
            counters[pipeline] = counters.get(pipeline, 0) + 1
            numbered.append((pipeline, counters[pipeline], event, item))
    numbered.sort(key=lambda n: (n[2].start, n[0], n[1]))

    rows = []
    for pipeline, number, event, item in numbered:
        model = _model(item.result, item.contents)
        version = _version(item.result)
        for measure in event.measures:
            box = event.box if measure.measure in ("person", "face") else None
            rows.append(
                Row(
                    run=video.run,
                    video=video.video,
                    pipeline=pipeline,
                    event=number,
                    track=event.track,
                    start=event.start,
                    end=event.end,
                    frame=event.frame,
                    measure=measure.measure,
                    label=measure.label,
                    value=measure.value,
                    confidence=measure.confidence,
                    x=box[0] if box else None,
                    y=box[1] if box else None,
                    width=box[2] if box else None,
                    height=box[3] if box else None,
                    model=model,
                    videoannotator_version=version,
                    file=item.result.path.name,
                )
            )
    return rows


def notes_path(video: VideoResults) -> str | None:
    """Where a video's notes are filed: the video the run read."""
    return video.media_path or video.video_path


def note_rows(notes: list[Any], video: str) -> list[Row]:
    """A video's notes as table rows (contracts/table-v2.md): run empty, pipeline
    `notes`, numbered within the video in order."""
    from ..version import __version__

    return [
        Row(
            run="",
            video=video,
            pipeline=NOTES_PIPELINE,
            event=number,
            track="",
            start=note.start_sec,
            end=note.end_sec,
            frame=None,
            measure=NOTE_MEASURE,
            label=note.text,
            value=None,
            confidence=None,
            x=None,
            y=None,
            width=None,
            height=None,
            model="",
            videoannotator_version=__version__,
            file="",
            author=note.author_name,
        )
        for number, note in enumerate(notes, 1)
    ]


class _Notes:
    """Each video's notes once per export, however many runs of it it covers."""

    def __init__(self, notes_for: NotesFor | None, summary: ExportSummary):
        self.notes_for = notes_for
        self.summary = summary
        self.seen: set[str] = set()

    def rows_for(self, video: VideoResults) -> list[Row]:
        path = notes_path(video)
        if self.notes_for is None or path is None:
            return []
        try:
            notes = self.notes_for(video)
        except NotesUnavailable:
            self.notes_for = None
            self.summary.notes.append(NO_NOTES_DATABASE)
            return []
        if not notes:
            return []
        key = notes[0].video_path
        if key in self.seen:
            return []
        self.seen.add(key)
        self.summary.researcher_notes += len(notes)
        return note_rows(notes, Path(path).stem)


def rows(
    results: ResultsSet,
    summary: ExportSummary | None = None,
    notes_for: NotesFor | None = None,
) -> Iterator[Row]:
    """Every row of `results`, video by video; fills in `summary` as it goes.
    A video's notes (`notes_for`) come once, before its first run's rows."""
    summary = summary if summary is not None else ExportSummary()
    notes = _Notes(notes_for, summary)
    pipelines: set[str] = set()
    for video in results.videos:
        researcher = notes.rows_for(video)
        if researcher:
            summary.rows += len(researcher)
            pipelines.add(NOTES_PIPELINE)
            yield from researcher
        read = read_video(video)
        video_table = video_rows(video, read)
        summary.sources += [
            {
                "run": video.run,
                "video": video.video,
                "file": item.result.path.name,
                "provenance": item.result.provenance,
            }
            for item in read
        ]
        if video_table:
            summary.videos += 1
            summary.rows += len(video_table)
            pipelines.update(r.pipeline for r in video_table)
        yield from video_table
    summary.pipelines = sorted(pipelines)
    summary.skipped = [s.to_dict() for s in results.skipped]
    summary.missing = [m.to_dict() for m in results.missing]
    summary.notes = list(results.notes) + summary.notes


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_csv(
    results: ResultsSet, out: TextIO, notes_for: NotesFor | None = None
) -> ExportSummary:
    """Write the table to `out` (opened with newline="")."""
    summary = ExportSummary()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows(results, summary, notes_for):
        writer.writerow([_cell(v) for v in row.values()])
    return summary


def parquet_available() -> bool:
    try:
        import pyarrow
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        return False
    return True


_INTEGER_COLUMNS = {"event", "frame"}
_NUMBER_COLUMNS = {"start", "end", "value", "confidence", "x", "y", "width", "height"}


def write_parquet(
    results: ResultsSet, path: str | Path, notes_for: NotesFor | None = None
) -> ExportSummary:
    """Write the table to a Parquet file at `path`."""
    if not parquet_available():
        raise ParquetNotInstalled()
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            (
                c,
                pa.int64()
                if c in _INTEGER_COLUMNS
                else pa.float64()
                if c in _NUMBER_COLUMNS
                else pa.string(),
            )
            for c in COLUMNS
        ]
    )
    summary = ExportSummary()
    batch: list[Row] = []

    def flush(writer: Any) -> None:
        columns = {c: [getattr(r, c) for r in batch] for c in COLUMNS}
        for c in _NUMBER_COLUMNS:
            columns[c] = [None if v is None else float(v) for v in columns[c]]
        for c in COLUMNS:
            if c not in _NUMBER_COLUMNS and c not in _INTEGER_COLUMNS:
                columns[c] = [None if v is None else str(v) for v in columns[c]]
        writer.write_table(pa.Table.from_pydict(columns, schema=schema))
        batch.clear()

    with pq.ParquetWriter(str(path), schema) as writer:
        for row in rows(results, summary, notes_for):
            batch.append(row)
            if len(batch) >= 50_000:
                flush(writer)
        if batch or summary.rows == 0:
            flush(writer)
        writer.add_key_value_metadata(
            {
                "videoannotator.table_format_version": str(TABLE_FORMAT_VERSION),
                "videoannotator.export_summary": json.dumps(
                    summary.to_dict(sources=True)
                ),
            }
        )
    return summary


def vlm_coco_from_answers(
    answers: list[dict[str, Any]], *, video_stem: str, model: str
) -> dict[str, Any]:
    """Ask AI try answers as the COCO file a vlm_annotation run writes (spec 027
    R16), so a try's table has a run's columns and measures. Skipped samples
    aren't annotations, as in a run; a model error is labelled as a run labels it.
    """
    annotations = []
    images = []
    for number, answer in enumerate(answers, 1):
        status = answer.get("status")
        if status == "skipped":
            continue
        frames = answer.get("frame_numbers") or [0]
        center = int(frames[len(frames) // 2])
        image_id = f"{video_stem}_frame_{center:06d}"
        label = (
            f"ERROR: {answer.get('error')}"
            if status == "error"
            else answer.get("label")
        )
        annotations.append(
            {
                "id": number,
                "image_id": image_id,
                "category_id": 1,
                "timestamp_sec": answer.get("t"),
                "frame_number": center,
                "context_frame_numbers": frames,
                "frame_times": answer.get("frame_times") or [],
                "status": status,
                "label": label,
                "reasoning": answer.get("reasoning"),
                "raw_response": answer.get("raw_response"),
                "model": model,
                "backend": "ollama",
            }
        )
        images.append(
            {"id": image_id, "frame_number": center, "timestamp": answer.get("t")}
        )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "vlm_annotation"}],
    }


def write_vlm_answers_csv(
    answers: list[dict[str, Any]], *, video_stem: str, model: str, out: TextIO
) -> ExportSummary:
    """A try's table, through the same reader and row builder as a run's."""
    import tempfile

    from .results_set import from_folder

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{video_stem}_vlm_annotation.json"
        path.write_text(
            json.dumps(
                vlm_coco_from_answers(answers, video_stem=video_stem, model=model)
            ),
            encoding="utf-8",
        )
        return write_csv(from_folder(folder), out)
