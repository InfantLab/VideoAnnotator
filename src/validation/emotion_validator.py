"""Emotion output schema validation utilities (v0.1).

Validates JSON structures conforming to docs/specs/emotion_output_format.md.

Design Goals:
- Lightweight (no external deps)
- Non-fatal: return structured errors; callers decide to raise
- Extensible: add new rule functions without breaking public API

Public API:
- validate_emotion_data(data: dict) -> List[str]
- validate_emotion_file(path: Path | str) -> List[str]

Return value: list of error strings. Empty list means valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = ["schema_version", "source_pipeline", "emotions"]


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def validate_emotion_data(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    # Basic presence
    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            _err(errors, f"Missing top-level field: {key}")
    # Do NOT early-return; accumulate additional type errors for better diagnostics

    # Types
    if not isinstance(data.get("schema_version"), int):
        _err(errors, "schema_version must be int")
    if not isinstance(data.get("source_pipeline"), str):
        _err(errors, "source_pipeline must be str")

    emotions = data.get("emotions")
    if not isinstance(emotions, list) or len(emotions) == 0:
        _err(errors, "emotions must be a non-empty list")
        # Continue collecting other top-level type errors but skip per-entry validation
        return errors

    for idx, entry in enumerate(emotions):
        prefix = f"emotions[{idx}]"
        if not isinstance(entry, dict):
            _err(errors, f"{prefix} not an object")
            continue
        # Required fields
        for k in ("start", "end", "labels"):
            if k not in entry:
                _err(errors, f"{prefix} missing field '{k}'")
        # Skip deeper checks if missing basics
        if any(k not in entry for k in ("start", "end", "labels")):
            continue
        start = entry["start"]
        end = entry["end"]
        if not isinstance(start, (int, float)):
            _err(errors, f"{prefix}.start must be number")
        if not isinstance(end, (int, float)):
            _err(errors, f"{prefix}.end must be number")
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and not (end > start)
        ):
            _err(errors, f"{prefix}.end must be greater than start")
        labels = entry["labels"]
        if not isinstance(labels, list) or len(labels) == 0:
            _err(errors, f"{prefix}.labels must be non-empty list")
        else:
            seen = set()
            top_conf = None
            for li, lab in enumerate(labels):
                if not isinstance(lab, dict):
                    _err(errors, f"{prefix}.labels[{li}] not an object")
                    continue
                label_name = lab.get("label")
                conf = lab.get("confidence")
                if not isinstance(label_name, str):
                    _err(errors, f"{prefix}.labels[{li}].label must be str")
                if not isinstance(conf, (int, float)):
                    _err(errors, f"{prefix}.labels[{li}].confidence must be number")
                else:
                    if not (0.0 <= conf <= 1.0):
                        _err(
                            errors,
                            f"{prefix}.labels[{li}].confidence out of range [0,1]",
                        )
                if label_name in seen:
                    _err(errors, f"{prefix}.labels duplicate label '{label_name}'")
                else:
                    seen.add(label_name)
                if li == 0:
                    top_conf = conf
            # Optional duplicate convenience confidence
            if "confidence" in entry:
                if not isinstance(entry["confidence"], (int, float)):
                    _err(errors, f"{prefix}.confidence must be number if present")
                elif (
                    isinstance(top_conf, (int, float))
                    and abs(entry["confidence"] - top_conf) > 1e-6
                ):
                    _err(
                        errors,
                        f"{prefix}.confidence should match first label confidence",
                    )
        # source object
        source = entry.get("source")
        if source is not None:
            if not isinstance(source, dict):
                _err(errors, f"{prefix}.source must be object")
            else:
                modality = source.get("modality")
                if modality is None:
                    _err(
                        errors, f"{prefix}.source.modality required when source present"
                    )
                elif modality not in ("video", "audio", "multimodal"):
                    _err(errors, f"{prefix}.source.modality invalid: {modality}")
        # quality_flags
        qf = entry.get("quality_flags")
        if qf is not None:
            if not isinstance(qf, list) or not all(isinstance(x, str) for x in qf):
                _err(errors, f"{prefix}.quality_flags must be list[str]")
    return errors


def validate_emotion_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return [f"File not found: {p}"]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # broad: return error to caller
        return [f"Failed to parse JSON: {e}"]
    return validate_emotion_data(data)


__all__ = ["validate_emotion_data", "validate_emotion_file"]
