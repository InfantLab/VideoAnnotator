#!/usr/bin/env python
"""Generate pipelines_spec.md from registry metadata.

Usage (local):
    uv run python scripts/generate_pipeline_specs.py

Typical CI integration (pseudo):
    uv run python scripts/generate_pipeline_specs.py
    git diff --exit-code docs/pipelines_spec.md  # fail if uncommitted drift

Purpose:
    Single authoritative markdown summary of all registered pipelines. Downstream
    docs and release notes MUST link to this instead of duplicating tables.

When to re-run:
    - Adding a new YAML under src/registry/metadata/
    - Modifying taxonomy fields / outputs / config_schema of an existing pipeline
    - Changing ordering logic here (rare)

Notes:
    - This file is intentionally ASCII-only (Windows console safety).
    - Regeneration is idempotent aside from the timestamp line.
    - Keep the column list (COLUMNS) minimal in v1.2.x; richer descriptors belong to v1.3.x+.
"""

from datetime import datetime
from pathlib import Path

from videoannotator.registry.pipeline_registry import get_registry

OUTPUT_PATH = Path("docs/pipelines_spec.md")

COLUMNS = [
    ("Name", lambda m: m.name),
    ("Display Name", lambda m: m.display_name),
    ("Family", lambda m: m.pipeline_family or "-"),
    ("Variant", lambda m: m.variant or "-"),
    ("Tasks", lambda m: ",".join(m.tasks) if m.tasks else "-"),
    ("Modalities", lambda m: ",".join(m.modalities) if m.modalities else "-"),
    ("Capabilities", lambda m: ",".join(m.capabilities) if m.capabilities else "-"),
    ("Backends", lambda m: ",".join(m.backends) if m.backends else "-"),
    ("Stability", lambda m: m.stability or "-"),
    (
        "Outputs",
        lambda m: ";".join(f"{o.format}:{'/'.join(o.types)}" for o in m.outputs),
    ),
]


def generate():
    reg = get_registry()
    reg.load(force=True)
    metas = sorted(reg.list(), key=lambda m: m.name)
    # Basic markdown table
    header = "| " + " | ".join(col for col, _ in COLUMNS) + " |\n"
    sep = "| " + " | ".join(["---"] * len(COLUMNS)) + " |\n"
    rows = []
    for m in metas:
        rows.append("| " + " | ".join(fn(m) for _, fn in COLUMNS) + " |")
    content = [
        "# Pipeline Specifications",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "This file is auto-generated. Do not edit by hand.",
        "",
        header + sep + "\n".join(rows),
        "",
        "Total pipelines: " + str(len(metas)),
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    generate()
    print(f"[OK] Wrote {OUTPUT_PATH}")
