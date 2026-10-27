"""Answer shapes against the cases the viewer's twin also reads (spec 033)."""

import json
from pathlib import Path

import pytest

from videoannotator.pipelines.vlm_annotation import answer_shape as shapes

CASES = json.loads(
    (
        Path(__file__).parent.parent / "fixtures" / "answer_shape" / "cases.json"
    ).read_text(encoding="utf-8")
)
VALID = {c["name"]: c for c in CASES["shapes"]["valid"]}


@pytest.mark.parametrize("case", CASES["shapes"]["valid"], ids=lambda c: c["name"])
def test_valid_shapes_and_their_schema(case):
    shape = shapes.validate_shape(case["shape"])
    assert shapes.shape_dicts(shape) == case["shape"]
    assert shapes.to_json_schema(shape) == case["json_schema"]


@pytest.mark.parametrize("case", CASES["shapes"]["invalid"], ids=lambda c: c["name"])
def test_invalid_shapes_name_the_problem(case):
    with pytest.raises(ValueError) as e:
        shapes.validate_shape(case["shape"])
    assert str(e.value) == case["problem"]


@pytest.mark.parametrize("case", CASES["replies"], ids=lambda c: c["name"])
def test_replies(case):
    shape = shapes.validate_shape(VALID[case["shape"]]["shape"])
    result = shapes.check_reply(shape, case["text"])
    assert {
        "fields": result.fields,
        "fit": result.fit,
        "fit_problem": result.fit_problem,
        "label": result.label,
    } == case["expect"]


@pytest.mark.parametrize("case", CASES["suggest"], ids=lambda c: c["name"])
def test_suggest_by_rule(case):
    got = shapes.suggest_by_rule(case["prompt"])
    assert (shapes.shape_dicts(got) if got else None) == case["expect"]


def test_the_hash_is_stable_across_key_order():
    a = [{"name": "score", "kind": "number", "min": 1, "max": 5}]
    b = [{"max": 5, "kind": "number", "min": 1, "name": "score"}]
    assert shapes.shape_sha256(shapes.validate_shape(a)) == shapes.shape_sha256(
        shapes.validate_shape(b)
    )
    c = [{"name": "score", "kind": "number", "min": 1, "max": 7}]
    assert shapes.shape_sha256(shapes.validate_shape(a)) != shapes.shape_sha256(
        shapes.validate_shape(c)
    )


def test_the_prompt_text_for_a_server_that_cant_enforce_it():
    schema = shapes.to_json_schema(
        shapes.validate_shape(VALID["yes/no, choice, number 1-5"]["shape"])
    )
    assert shapes.describe_for_prompt(schema) == (
        'Answer with JSON only, one object with exactly these keys: {"touch": "yes" '
        'or "no", "initiator": "mother" or "infant" or "both" or "none", '
        '"confidence": a number from 1 to 5}'
    )


@pytest.mark.parametrize(
    ("prompt", "fields"),
    [
        (
            "How many adults are in the room, and is the child smiling?",
            [("Adults", "number"), ("Child smiling", "yes_no")],
        ),
        (
            "Is the baby crying? How many toys can you see?",
            [("Baby crying", "yes_no"), ("Toys", "number")],
        ),
        ("What is the mother doing with her hands?", None),
    ],
)
def test_a_question_of_several_parts_gets_a_field_per_part(prompt, fields):
    found = shapes.suggest_by_rule(prompt)
    assert ([(f.name, f.kind) for f in found] if found else None) == fields
