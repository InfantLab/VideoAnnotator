"""Where a VLM run samples, as plain functions (spec 027 research R1-R3).

One definition used by runs, tries and the frames endpoint; the viewer's
`lib/sampling.ts` is its twin, and both are pinned by
`tests/fixtures/ask_ai/sampling_cases.json` so the ticks the window draws are
the points the pipeline samples.
"""

from __future__ import annotations

import math

# Floating-point slack so a grid point that lands on `end` by arithmetic noise
# (0.1 + 0.2 > 0.3) is still sampled.
_EPS = 1e-9


def _round3(t: float) -> float:
    return round(t, 3)


def _half_up(x: float) -> int:
    """Round half up: Python's round() is half-to-even and JavaScript's
    Math.round() is half-up, so the twin uses this in both."""
    return math.floor(x + 0.5)


def sample_points(
    duration: float,
    interval: float,
    start: float = 0.0,
    end: float | None = None,
    moments: list[float] | None = None,
) -> list[float]:
    """Listed moments within the video when given, else the interval grid."""
    if moments is not None:
        kept = {_round3(t) for t in moments if 0 <= t <= duration + _EPS}
        return sorted(kept)
    if interval <= 0:
        raise ValueError("interval must be greater than 0")
    stop = duration if end is None else min(end, duration)
    points: list[float] = []
    k = 0
    while True:
        t = start + k * interval
        if t > stop + _EPS:
            break
        points.append(_round3(t))
        k += 1
    return points


def burst_offsets_for(frames: int) -> list[int]:
    """`frames` consecutive offsets around 0; an even count has one more
    before the anchor than after (4 -> [-2, -1, 0, 1])."""
    before = frames // 2
    return list(range(-before, frames - before))


def burst_times(
    anchor: float, offsets: list[int], spacing: float, duration: float
) -> list[tuple[int, float]]:
    """(offset, time) of each burst frame inside the video."""
    out: list[tuple[int, float]] = []
    for off in offsets:
        t = anchor + off * spacing
        if t < -_EPS or t > duration + _EPS:
            continue
        out.append((off, _round3(max(0.0, t))))
    return out


def choose_moments(points: list[float], count: int) -> list[float]:
    """`count` points spread evenly over `points` (research R3)."""
    s = len(points)
    if count <= 0 or s == 0:
        return []
    if s <= count:
        return list(points)
    if count == 1:
        return [points[_half_up((s - 1) / 2)]]
    return [points[_half_up(i * (s - 1) / (count - 1))] for i in range(count)]


def clip_span(anchor: float, length: float, duration: float) -> tuple[float, float]:
    """[start, end] of a clip centred on `anchor`, shortened (not shifted) at
    the video's ends (spec 033 R12)."""
    half = length / 2
    return _round3(max(0.0, anchor - half)), _round3(min(duration, anchor + half))


def clip_times(
    anchor: float, length: float, fps: float, duration: float
) -> list[float]:
    """The times of a clip's frames: from its start, `fps` a second, up to
    but not including its end; at least the start."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    start, end = clip_span(anchor, length, duration)
    times: list[float] = []
    k = 0
    while start + k / fps <= end - _EPS:
        times.append(_round3(start + k / fps))
        k += 1
    return times or [start]
