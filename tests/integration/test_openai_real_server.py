"""The demo clip through a real OpenAI-compatible server (spec 033 R15).

Opt-in: skipped unless VA_TEST_OPENAI_BASE_URL and VA_TEST_OPENAI_MODEL name a
server on this computer or the local network, e.g.

    VA_TEST_OPENAI_BASE_URL=http://127.0.0.1:8080/v1 VA_TEST_OPENAI_MODEL=gemma-3-4b \\
        pytest -m real_vlm_server tests/integration/test_openai_real_server.py
"""

import json
import os
from pathlib import Path

import pytest

from videoannotator.endpoints import classify_endpoint
from videoannotator.pipelines.vlm_annotation import vlm_pipeline

BASE_URL = os.environ.get("VA_TEST_OPENAI_BASE_URL", "")
MODEL = os.environ.get("VA_TEST_OPENAI_MODEL", "")
DEMO = next(
    iter(sorted((Path(__file__).parents[2] / "viewer" / "demo-assets").glob("*.mp4"))),
    None,
)

pytestmark = [
    pytest.mark.real_vlm_server,
    pytest.mark.skipif(
        not (BASE_URL and MODEL),
        reason="set VA_TEST_OPENAI_BASE_URL and VA_TEST_OPENAI_MODEL",
    ),
    pytest.mark.skipif(DEMO is None, reason="the demo clip isn't here"),
]

SHAPE = [
    {"name": "touch", "kind": "yes_no"},
    {"name": "people", "kind": "number", "min": 0, "max": 10},
]


def _run(tmp_path, **config):
    assert classify_endpoint(BASE_URL).cls == "local", "use a local server"
    p = vlm_pipeline.VLMAnnotationPipeline(
        {
            "backend": "openai_compatible",
            "base_url": BASE_URL,
            "model": MODEL,
            "frame_interval_sec": 4.0,
            **config,
        }
    )
    p.initialize()
    p.process(str(DEMO), output_dir=str(tmp_path))
    (out,) = tmp_path.glob("*_vlm_annotation.json")
    return p, json.loads(out.read_text())


def test_free_text(tmp_path):
    p, data = _run(tmp_path)
    assert data["annotations"]
    assert {a["backend"] for a in data["annotations"]} == {"openai_compatible"}
    assert all(a["status"] == "answered" for a in data["annotations"])
    assert p.provenance_vlm()["backend"] == "openai_compatible"
    (ref,) = p.provenance_models()
    assert ref.source == "openai_compatible"


def test_shaped(tmp_path):
    p, data = _run(tmp_path, answer_shape=SHAPE)
    annotations = data["annotations"]
    fits = [a["fit"] for a in annotations]
    assert all(isinstance(f, bool) for f in fits)
    print(
        f"fit rate {sum(fits)}/{len(fits)}; {p.run_summary().get('structured_output')}"
    )
    assert p.run_summary()["structured_output"] in ("enforced", "prompt_only")
    assert data["vlm_answer_shape"] == SHAPE
