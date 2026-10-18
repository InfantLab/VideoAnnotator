"""Compare two kept demo-clip runs, file by file (spec 030 quickstart §2).

    python scripts/compare_demo_outputs.py <before> <after>

Each folder holds the files `tests/integration/test_output_baseline.py` keeps when
VIDEOANNOTATOR_BASELINE_KEEP is set. Provenance and creation times are ignored, as in
that test. Prints per file: identical, within the baseline test's tolerance, or
different, with a short summary of what changed for the formats a default change
touches.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

VTT_MARKER = "NOTE videoannotator-provenance"
PREFIX = "demo_clip_"
# Same bounds as tests/integration/test_output_baseline.py's TOLERANCE.
TOLERANCE = {
    "person_tracking.json": (1e-2, 1e-3),
    "openface3_analysis.json": (5e-2, 1e-2),
    "openface3_detailed.json": (5e-2, 1e-2),
    "face_detections.json": (1e-2, 1e-3),
}


def normalise(suffix: str, text: str) -> Any:
    """The annotations only: provenance and creation times differ every run."""
    if suffix.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, dict):
            data.pop("provenance", None)
            if isinstance(data.get("info"), dict):
                data["info"].pop("date_created", None)
                data["info"].pop("year", None)
                # The release that wrote it; provenance says the same.
                data["info"].pop("version", None)
        return data
    if suffix.endswith(".vtt"):
        text = "\n\n".join(
            b for b in text.split("\n\n") if not b.startswith(VTT_MARKER)
        )
    return text.rstrip("\n")


def within(current: Any, golden: Any, rel: float, abs_: float) -> bool:
    if isinstance(golden, dict):
        return (
            isinstance(current, dict)
            and current.keys() == golden.keys()
            and all(within(current[k], golden[k], rel, abs_) for k in golden)
        )
    if isinstance(golden, list):
        return (
            isinstance(current, list)
            and len(current) == len(golden)
            and all(
                within(c, g, rel, abs_) for c, g in zip(current, golden, strict=True)
            )
        )
    if isinstance(golden, int | float) and not isinstance(golden, bool):
        if not isinstance(current, int | float) or isinstance(current, bool):
            return False
        return math.isclose(current, golden, rel_tol=rel, abs_tol=abs_)
    return current == golden


def _seconds(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def vtt_cues(text: str) -> list[tuple[float, float, str]]:
    cues = []
    for block in text.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            if "-->" in line:
                start, end = (part.strip() for part in line.split("-->"))
                cues.append((_seconds(start), _seconds(end), " ".join(lines[i + 1 :])))
                break
    return cues


def summarise_vtt(before: str, after: str) -> list[str]:
    a, b = vtt_cues(before), vtt_cues(after)
    matcher = difflib.SequenceMatcher(a=[c[2] for c in a], b=[c[2] for c in b])
    added = removed = reworded = 0
    shift = 0.0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for x, y in zip(a[i1:i2], b[j1:j2], strict=True):
                shift = max(shift, abs(x[0] - y[0]), abs(x[1] - y[1]))
        elif op == "replace":
            reworded += min(i2 - i1, j2 - j1)
            added += max(0, (j2 - j1) - (i2 - i1))
            removed += max(0, (i2 - i1) - (j2 - j1))
        elif op == "insert":
            added += j2 - j1
        elif op == "delete":
            removed += i2 - i1
    lines = [
        f"cues {len(a)} -> {len(b)}: {reworded} reworded, {added} added, "
        f"{removed} removed; largest timing shift of an unchanged cue {shift:.3f} s"
    ]
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "equal":
            lines.append(f"  - {[c[2] for c in a[i1:i2]]}")
            lines.append(f"  + {[c[2] for c in b[j1:j2]]}")
    return lines


def rttm_turns(text: str) -> list[tuple[float, float, str]]:
    turns = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            turns.append((float(parts[3]), float(parts[4]), parts[7]))
    return turns


def summarise_rttm(before: str, after: str) -> list[str]:
    lines = []
    for label, text in (("before", before), ("after", after)):
        turns = rttm_turns(text)
        speakers = Counter(t[2] for t in turns)
        speech = sum(t[1] for t in turns)
        lines.append(
            f"{label}: {len(turns)} turns, {len(speakers)} speakers "
            f"{dict(speakers)}, {speech:.2f} s of speech"
        )
    return lines


def summarise_coco(before: dict, after: dict) -> list[str]:
    lines = []
    for label, data in (("before", before), ("after", after)):
        anns = data.get("annotations", [])
        images = data.get("images", [])
        tracks = {a.get("track_id") for a in anns if a.get("track_id") is not None}
        keypoints = Counter(
            len(a["keypoints"]) // 3 for a in anns if a.get("keypoints")
        )
        keys = sorted({k for a in anns for k in a})
        line = f"{label}: {len(anns)} annotations over {len(images)} frames"
        if tracks:
            line += f", {len(tracks)} tracks"
        if keypoints:
            line += f", keypoints per annotation {dict(keypoints)}"
        lines.append(line)
        lines.append(f"  keys: {', '.join(keys)}")
    return lines


def compare(before_dir: Path, after_dir: Path) -> list[str]:
    report = []
    names = sorted(
        {p.name for p in before_dir.glob(f"{PREFIX}*")}
        | {p.name for p in after_dir.glob(f"{PREFIX}*")}
    )
    for name in names:
        suffix = name.removeprefix(PREFIX)
        before_path, after_path = before_dir / name, after_dir / name
        if not before_path.exists() or not after_path.exists():
            where = "after" if after_path.exists() else "before"
            report.append(f"{suffix}: only in {where}")
            continue
        before_text, after_text = before_path.read_text(), after_path.read_text()
        a, b = normalise(suffix, before_text), normalise(suffix, after_text)
        if a == b:
            report.append(f"{suffix}: identical")
            continue
        if suffix in TOLERANCE and within(b, a, *TOLERANCE[suffix]):
            report.append(f"{suffix}: within tolerance {TOLERANCE[suffix]}")
            continue
        report.append(f"{suffix}: different")
        if suffix.endswith(".vtt"):
            details = summarise_vtt(a, b)
        elif suffix.endswith(".rttm"):
            details = summarise_rttm(a, b)
        elif isinstance(a, dict) and "annotations" in a:
            details = summarise_coco(a, b)
        else:
            details = []
        report.extend(f"    {line}" for line in details)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    print("\n".join(compare(args.before.expanduser(), args.after.expanduser())))


if __name__ == "__main__":
    main()
