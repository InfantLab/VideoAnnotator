#!/usr/bin/env python3
"""Run pipelines on a video and compare their outputs between environments.

Used to check that a change (a Python version, a library upgrade) leaves pipeline
outputs unchanged: dump the outputs in each environment, then compare the dumps.
`videoannotator process` would be the natural tool but isn't implemented yet.

    python scripts/compare_pipeline_outputs.py dump VIDEO --pipelines a,b -o out.json
    python scripts/compare_pipeline_outputs.py compare before.json after.json [--rtol 1e-6]

`dump` runs one job through the API server's own JobProcessor, with a throwaway
SQLite store, and loads `.env` like the server does (speaker_diarization needs
HUGGINGFACE_TOKEN). `compare` exits 1 if any pipeline's status or annotations
differ, ignoring fields that change on every run.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Fields that differ between runs of the same job. `timestamp` is deliberately not
# here: in annotations it is the position in the video.
RUN_SPECIFIC_KEYS = {
    "id",
    "job_id",
    "annotation_id",
    "created_at",
    "date_created",
    "processing_time",
    "processing_timestamp",
    "total_time",
}
OUTPUT_DIR_PLACEHOLDER = "<OUTPUT_DIR>"


def dump(video: Path, pipelines: list[str], out: Path) -> int:
    from videoannotator.api.job_processor import JobProcessor
    from videoannotator.batch.types import BatchJob
    from videoannotator.config_env import load_env_file
    from videoannotator.storage.sqlite_backend import SQLiteStorageBackend

    load_env_file()
    with tempfile.TemporaryDirectory(prefix="va-compare-") as tmp:
        work = Path(tmp)
        storage = SQLiteStorageBackend(work / "jobs.db")
        job = BatchJob(
            video_path=video, output_dir=work / "out", selected_pipelines=pipelines
        )
        started = time.time()
        job = JobProcessor().process_job(job, storage)
        result: dict[str, Any] = {
            "video": str(video),
            "python": platform.python_version(),
            "job_status": str(job.status),
            "seconds": round(time.time() - started, 1),
            "pipelines": {},
        }
        for name in pipelines:
            pipeline_result = job.pipeline_results.get(name)
            annotations = (
                storage.load_annotations(job.job_id, name)
                if storage.annotation_exists(job.job_id, name)
                else []
            )
            result["pipelines"][name] = {
                "status": str(pipeline_result.status) if pipeline_result else "missing",
                "error": (pipeline_result.error_message if pipeline_result else None),
                "annotations": _normalise(annotations, str(work)),
            }
        storage.close()

    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    for name, p in result["pipelines"].items():
        print(f"{name}: {p['status']}, {len(p['annotations'])} annotations")
    print(f"Python {result['python']}, {result['seconds']}s -> {out}")
    return 0


def _normalise(value: Any, workdir: str) -> Any:
    if isinstance(value, dict):
        return {
            k: _normalise(v, workdir)
            for k, v in value.items()
            if k not in RUN_SPECIFIC_KEYS
        }
    if isinstance(value, list):
        return [_normalise(v, workdir) for v in value]
    if isinstance(value, str) and workdir in value:
        return value.replace(workdir, OUTPUT_DIR_PLACEHOLDER)
    return value


def _differences(a: Any, b: Any, rtol: float, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = [f"{path}.{k}: only in first" for k in a.keys() - b.keys()]
        diffs += [f"{path}.{k}: only in second" for k in b.keys() - a.keys()]
        for k in sorted(a.keys() & b.keys()):
            diffs += _differences(a[k], b[k], rtol, f"{path}.{k}")
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items vs {len(b)}"]
        diffs: list[str] = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diffs += _differences(x, y, rtol, f"{path}[{i}]")
        return diffs
    if (
        isinstance(a, int | float)
        and isinstance(b, int | float)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return (
            []
            if math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)
            else [f"{path}: {a!r} vs {b!r}"]
        )
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def compare(first: Path, second: Path, rtol: float) -> int:
    a = json.loads(first.read_text())
    b = json.loads(second.read_text())
    print(f"{first} (Python {a['python']}) vs {second} (Python {b['python']})")
    failed = False
    for name in sorted(a["pipelines"].keys() | b["pipelines"].keys()):
        pa, pb = a["pipelines"].get(name), b["pipelines"].get(name)
        if pa is None or pb is None:
            print(f"  {name}: only in {'second' if pa is None else 'first'}")
            failed = True
            continue
        diffs = _differences(
            {"status": pa["status"], "annotations": pa["annotations"]},
            {"status": pb["status"], "annotations": pb["annotations"]},
            rtol,
        )
        count = len(pa["annotations"])
        if diffs:
            failed = True
            print(
                f"  {name}: {len(diffs)} differences ({count} vs {len(pb['annotations'])} annotations)"
            )
            for d in diffs[:10]:
                print(f"    {d}")
            if len(diffs) > 10:
                print(f"    ... and {len(diffs) - 10} more")
        else:
            print(f"  {name}: identical ({pa['status']}, {count} annotations)")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="run pipelines on a video and save their outputs")
    d.add_argument("video", type=Path)
    d.add_argument("--pipelines", required=True, help="comma-separated pipeline names")
    d.add_argument("-o", "--output", type=Path, required=True)
    c = sub.add_parser("compare", help="compare two dumps")
    c.add_argument("first", type=Path)
    c.add_argument("second", type=Path)
    c.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="relative tolerance for numbers (default 0: exact)",
    )
    args = parser.parse_args()
    if args.command == "dump":
        return dump(args.video, args.pipelines.split(","), args.output)
    return compare(args.first, args.second, args.rtol)


if __name__ == "__main__":
    sys.exit(main())
