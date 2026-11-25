"""Shorten Whisper's repetition loops in a transcript.

Whisper can decode the same word or phrase over and over, most of all on
repetitive child-directed speech, laughter and noise ("no" 240 times over 17 s
of a nom-nom game). Retrying at higher temperatures removes some loops but makes
every run differ, so decoding stays at one temperature and loops are cut here:
deterministically, and marked with "[…]" so a reader knows text was removed.
"""

import re
from typing import Any

MARK = "[…]"
_MAX_UNIT = 6


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


def _shorten_tokens(tokens: list[str], limit: int) -> tuple[list[int], list[int]]:
    """Indices of tokens to keep, and the kept indices after which text was cut.

    A unit of one to six tokens repeated more than `limit` times in a row
    keeps its first `limit` repetitions.
    """
    norms = [_norm(t) for t in tokens]
    keep: list[int] = []
    cuts: list[int] = []
    i = 0
    while i < len(tokens):
        step = 0
        for n in range(1, _MAX_UNIT + 1):
            unit = norms[i : i + n]
            if len(unit) < n or not all(unit):
                break
            count = 1
            while norms[i + count * n : i + (count + 1) * n] == unit:
                count += 1
            if count > limit:
                keep.extend(range(i, i + limit * n))
                cuts.append(i + limit * n - 1)
                step = count * n
                break
        if step:
            i += step
        else:
            keep.append(i)
            i += 1
    return keep, cuts


def _shorten_segment(segment: dict[str, Any], limit: int) -> dict[str, Any]:
    words = segment.get("words") or []
    if words:
        tokens = [w["raw"] for w in words]
    else:
        tokens = re.findall(r"\s*\S+", segment["text"])
    keep, cuts = _shorten_tokens(tokens, limit)
    if not cuts:
        return segment
    cut_after = set(cuts)
    text = "".join(tokens[k] + (f" {MARK}" if k in cut_after else "") for k in keep)
    shortened = dict(segment, text=text.strip(), repeats_shortened=True)
    if words:
        shortened["words"] = [words[k] for k in keep]
    return shortened


def shorten_repeats(segments: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Cut loops within segments, then runs of identical segments.

    Each segment has `start`, `end`, `text` and optionally `words`, each word
    with its `raw` text (leading space included, as faster-whisper gives it).
    A limit of 0 or less leaves the transcript as decoded.
    """
    if limit <= 0:
        return segments
    out: list[dict[str, Any]] = []
    run = 0
    for segment in (_shorten_segment(s, limit) for s in segments):
        key = _segment_key(segment)
        run = run + 1 if out and key and key == _segment_key(out[-1]) else 1
        if run <= limit:
            out.append(segment)
            continue
        last = out[-1]
        text = last["text"] if last["text"].endswith(MARK) else f"{last['text']} {MARK}"
        out[-1] = dict(last, end=segment["end"], text=text, repeats_shortened=True)
    return out


def _segment_key(segment: dict[str, Any]) -> str:
    return _norm(segment["text"].replace(MARK, "").replace(" ", ""))
