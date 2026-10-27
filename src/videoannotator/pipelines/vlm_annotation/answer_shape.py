"""Answer shapes: a question's answer as named fields (spec 033 R8, R11).

A shape is a flat, ordered list of fields, each a choice, yes/no, number or
text. It is sent to the server as JSON Schema, and every reply is checked here
too, by a validator small enough to keep exact for four kinds: a reply that
doesn't fit is recorded as such, never forced into a value. The viewer's
`lib/answerShape.ts` is the twin of `validate_shape` and `suggest_by_rule`;
both are pinned by `tests/fixtures/answer_shape/cases.json`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["choice", "yes_no", "number", "text"]
KINDS: tuple[Kind, ...] = ("choice", "yes_no", "number", "text")
RESERVED = frozenset({"label", "reasoning", "fit", "fit_problem", "t"})
MAX_FIELDS = 12
MAX_VALUES = 30
MAX_VALUE_LENGTH = 60
TEXT_MAX_LENGTH = 200
DIDNT_FIT = "DIDN'T FIT"

# Names as the researcher types them ("Child smiling"), the JSON key as is (D5, UAT #29).
_NAME = re.compile(r"[^\x00-\x1f\"\\\s](?:[^\x00-\x1f\"\\]{0,38}[^\x00-\x1f\"\\\s])?")


@dataclass(frozen=True)
class Field:
    name: str
    kind: Kind
    values: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "choice":
            out["values"] = list(self.values)
        if self.kind == "number":
            if self.min is not None:
                out["min"] = self.min
            if self.max is not None:
                out["max"] = self.max
        return out


@dataclass
class ShapeResult:
    fields: dict[str, Any] = field(default_factory=dict)
    fit: bool = False
    fit_problem: str | None = None
    label: str = DIDNT_FIT


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_shape(raw: Any) -> list[Field]:
    """The shape as fields; ValueError naming the field and the rule."""
    if not isinstance(raw, list):
        raise ValueError("The answer shape must be a list of fields.")
    if not raw:
        raise ValueError("Add at least one field.")
    if len(raw) > MAX_FIELDS:
        raise ValueError(f"An answer shape has at most {MAX_FIELDS} fields.")
    fields: list[Field] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Field {i}: not a field.")
        name = item.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(
                f"Field {i}: '{name}' isn't a usable name: give it a name of 1 to 40 "
                "characters, without quotes or backslashes."
            )
        if name.lower() in RESERVED:
            raise ValueError(f"Field {i}: '{name}' is reserved; choose another name.")
        if name.lower() in seen:
            raise ValueError(f"Field {i}: '{name}' is used twice.")
        seen.add(name.lower())
        kind = item.get("kind")
        where = f"Field {i} ({name})"
        if kind not in KINDS:
            raise ValueError(
                f"{where}: the kind must be choice, yes_no, number or text."
            )
        if kind == "choice":
            values = item.get("values")
            if not isinstance(values, list) or not all(
                isinstance(v, str) for v in values
            ):
                raise ValueError(
                    f"{where}: a choice needs 2 to {MAX_VALUES} different values."
                )
            values = [v.strip() for v in values]
            if any(not 1 <= len(v) <= MAX_VALUE_LENGTH for v in values):
                raise ValueError(
                    f"{where}: each value must be 1 to {MAX_VALUE_LENGTH} characters."
                )
            if not 2 <= len(set(values)) == len(values) <= MAX_VALUES:
                raise ValueError(
                    f"{where}: a choice needs 2 to {MAX_VALUES} different values."
                )
            fields.append(Field(name, kind, tuple(values)))
        elif kind == "number":
            lo, hi = item.get("min"), item.get("max")
            if (lo is not None and not _is_number(lo)) or (
                hi is not None and not _is_number(hi)
            ):
                raise ValueError(f"{where}: the lowest and highest must be numbers.")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(
                    f"{where}: the lowest number is more than the highest."
                )
            fields.append(Field(name, kind, min=lo, max=hi))
        else:
            fields.append(Field(name, kind))
    return fields


def shape_dicts(shape: list[Field]) -> list[dict[str, Any]]:
    return [f.to_dict() for f in shape]


def shape_sha256(shape: list[Field]) -> str:
    text = json.dumps(shape_dicts(shape), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json_schema(shape: list[Field]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for f in shape:
        if f.kind == "choice":
            properties[f.name] = {"type": "string", "enum": list(f.values)}
        elif f.kind == "yes_no":
            properties[f.name] = {"type": "string", "enum": ["yes", "no"]}
        elif f.kind == "number":
            prop: dict[str, Any] = {"type": "number"}
            if f.min is not None:
                prop["minimum"] = f.min
            if f.max is not None:
                prop["maximum"] = f.max
            properties[f.name] = prop
        else:
            properties[f.name] = {"type": "string", "maxLength": TEXT_MAX_LENGTH}
    return {
        "type": "object",
        "properties": properties,
        "required": [f.name for f in shape],
        "additionalProperties": False,
    }


def describe_for_prompt(schema: dict[str, Any]) -> str:
    """The shape in words, for a server that can't enforce it: added to the
    end of the prompt (R8)."""
    parts = []
    for name, prop in (schema.get("properties") or {}).items():
        if "enum" in prop:
            what = " or ".join(json.dumps(v) for v in prop["enum"])
        elif prop.get("type") == "number":
            lo, hi = prop.get("minimum"), prop.get("maximum")
            what = "a number"
            if lo is not None and hi is not None:
                what += f" from {_num_text(lo)} to {_num_text(hi)}"
            elif lo is not None:
                what += f" of at least {_num_text(lo)}"
            elif hi is not None:
                what += f" of at most {_num_text(hi)}"
        else:
            what = "a short text"
        parts.append(f'"{name}": {what}')
    return (
        "Answer with JSON only, one object with exactly these keys: {"
        + ", ".join(parts)
        + "}"
    )


def _num_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Checking a reply
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def first_object(text: str) -> dict[str, Any] | None:
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for start in (i for i, c in enumerate(candidate) if c == "{"):
            try:
                value, _ = decoder.raw_decode(candidate, start)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _check_field(f: Field, value: Any) -> tuple[Any, str | None]:
    if f.kind == "choice":
        if isinstance(value, str):
            for v in f.values:
                if value.strip().lower() == v.lower():
                    return v, None
        return None, f"{f.name}: {_shown(value)} is not one of {', '.join(f.values)}"
    if f.kind == "yes_no":
        if isinstance(value, bool):
            return ("yes" if value else "no"), None
        if isinstance(value, str) and value.strip().lower() in ("yes", "no"):
            return value.strip().lower(), None
        return None, f"{f.name}: {_shown(value)} is not one of yes, no"
    if f.kind == "number":
        if not _is_number(value):
            return None, f"{f.name}: {_shown(value)} is not a number"
        number = (
            int(value) if isinstance(value, float) and value.is_integer() else value
        )
        if (f.min is not None and number < f.min) or (
            f.max is not None and number > f.max
        ):
            lo = "" if f.min is None else _num_text(f.min)
            hi = "" if f.max is None else _num_text(f.max)
            return None, f"{f.name}: {_num_text(number)} is outside {lo} to {hi}"
        return number, None
    if not isinstance(value, str):
        return None, f"{f.name}: {_shown(value)} is not text"
    if len(value) > TEXT_MAX_LENGTH:
        return None, f"{f.name}: longer than {TEXT_MAX_LENGTH} characters"
    return value, None


def _shown(value: Any) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    return json.dumps(value)


def check_reply(shape: list[Field], text: str | None) -> ShapeResult:
    """The reply's fields, and whether it fits the shape (FR-020)."""
    if not text or not text.strip():
        return ShapeResult(fit_problem="The reply is empty.")
    obj = first_object(text)
    if obj is None:
        return ShapeResult(fit_problem="The reply isn't JSON.")
    fields: dict[str, Any] = {}
    problems: list[str] = []
    for f in shape:
        if f.name not in obj:
            problems.append(f"{f.name}: missing")
            continue
        value, problem = _check_field(f, obj[f.name])
        if problem:
            problems.append(problem)
        else:
            fields[f.name] = value
    known = {f.name for f in shape}
    problems += [f"unexpected field '{k}'" for k in obj if k not in known]
    if problems:
        return ShapeResult(fields=fields, fit_problem="; ".join(problems))
    first = fields[shape[0].name]
    return ShapeResult(
        fields=fields,
        fit=True,
        label=_num_text(first) if _is_number(first) else str(first),
    )


# ---------------------------------------------------------------------------
# Suggesting a shape from the question's text (R11)
# ---------------------------------------------------------------------------

_TRIGGER = (
    r"\b(?:return\s+only|(?:answer|reply|respond)\s+with(?:\s+only)?"
    r"(?:\s+one\s+(?:word|of))?|choose\s+(?:one\s+)?(?:of|from)|one\s+of)"
    r"\s*:?\s*"
)
_LIST_AFTER_TRIGGER = re.compile(_TRIGGER + r"([^\n.?!;()]+)", re.IGNORECASE)
_PARENTHESISED = re.compile(r"\(([^()\n]+)\)")
_SCALE = re.compile(
    r"\bscale\s+(?:of|from)\s+(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_YES_OR_NO = re.compile(r"\byes\s+or\s+no\b", re.IGNORECASE)
_A_NUMBER = re.compile(
    r"\b(?:answer|reply|respond)\s+with\s+(?:a|one)\s+(?:whole\s+)?number\b",
    re.IGNORECASE,
)
# Up to four words: plain values read better than underscored ones (UAT #58).
_ITEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*(?: [A-Za-z0-9][A-Za-z0-9_\-']*){0,3}")
_SPLIT = re.compile(r"\s*(?:,|/|\bor\b)\s*", re.IGNORECASE)


def _items(text: str) -> list[str] | None:
    items = [p for p in (s.strip() for s in _SPLIT.split(text)) if p]
    if not 2 <= len(items) <= MAX_VALUES or len(set(items)) != len(items):
        return None
    if not all(_ITEM.fullmatch(i) for i in items):
        return None
    return items


def _list_field(items: list[str]) -> dict[str, Any]:
    if {i.lower() for i in items} == {"yes", "no"} and len(items) == 2:
        return {"kind": "yes_no"}
    return {"kind": "choice", "values": items}


def _parse_number(text: str) -> int | float:
    value = float(text)
    return int(value) if value.is_integer() else value


def suggest_by_rule(prompt: str) -> list[Field] | None:
    """Fields for the lists of options the question itself gives, or None."""
    found: list[tuple[int, int, int, dict[str, Any]]] = []
    for m in _LIST_AFTER_TRIGGER.finditer(prompt):
        items = _items(m.group(1))
        if items:
            found.append((m.start(), m.end(), 0, _list_field(items)))
    for m in _PARENTHESISED.finditer(prompt):
        inner = m.group(1)
        if "/" not in inner and not re.search(r"\bor\b", inner, re.IGNORECASE):
            continue
        items = _items(inner)
        if items:
            found.append((m.start(), m.end(), 1, _list_field(items)))
    for m in _SCALE.finditer(prompt):
        lo, hi = _parse_number(m.group(1)), _parse_number(m.group(2))
        if lo <= hi:
            found.append(
                (m.start(), m.end(), 2, {"kind": "number", "min": lo, "max": hi})
            )
    for m in _YES_OR_NO.finditer(prompt):
        found.append((m.start(), m.end(), 3, {"kind": "yes_no"}))
    for m in _A_NUMBER.finditer(prompt):
        found.append((m.start(), m.end(), 4, {"kind": "number"}))

    kept: list[dict[str, Any]] = []
    end = -1
    for start, stop, _, spec in sorted(found, key=lambda x: (x[0], x[2])):
        if start < end:
            continue
        kept.append(spec)
        end = stop
    if not kept:
        return _by_parts(prompt)
    raw = [
        {"name": "Answer" if i == 0 else f"Answer {i + 1}", **spec}
        for i, spec in enumerate(kept[:MAX_FIELDS])
    ]
    return validate_shape(raw)


# A question of several parts, none listing its answers (UAT #57): "How many
# adults are in the room, and is the child smiling?" is a number and a yes or no.
_PARTS = re.compile(
    r"[?\n]+|[.;]\s+|,?\s+and\s+(?=(?:is|are|does|do|did|can|has|have|was|were|"
    r"how\s+many)\b)",
    re.IGNORECASE,
)
_HOW_MANY = re.compile(
    r"^how\s+many\s+([a-z' -]+?)(?:\s+(?:are|is|can|do|does|in|on)\b|$)", re.I
)
_YES_NO_Q = re.compile(r"^(?:is|are|does|do|did|can|has|have|was|were)\s+(.+)$", re.I)
_LEADING = re.compile(r"^(?:there\s+)?(?:the|a|an|any|some)\s+", re.I)
_QUESTION = re.compile(
    r"^(?:how|what|where|who|whom|which|why|when|is|are|does|do|did|can|has|have|was|were)\b",
    re.I,
)


def _label(words: str) -> str:
    text = _LEADING.sub("", words.strip(" ,.:")).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 40:
        text = text[:40].rsplit(" ", 1)[0]
    return text[:1].upper() + text[1:]


def _by_parts(prompt: str) -> list[Field] | None:
    """One field per part of the question, when every part says its kind; else None."""
    raw: list[dict[str, Any]] = []
    for part in (p.strip() for p in _PARTS.split(prompt)):
        if not part or not _QUESTION.match(part):
            continue
        if m := _HOW_MANY.match(part):
            raw.append({"name": _label(m.group(1)), "kind": "number"})
        elif m := _YES_NO_Q.match(part):
            raw.append({"name": _label(m.group(1)), "kind": "yes_no"})
        else:
            return None
    names = [f["name"].lower() for f in raw]
    if not raw or len(raw) > MAX_FIELDS or len(set(names)) != len(names) or "" in names:
        return None
    try:
        return validate_shape(raw)
    except ValueError:
        return None
