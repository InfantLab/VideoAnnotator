#!/usr/bin/env python
"""Generate pipelines_spec.md from registry metadata.

Usage: python scripts/generate_pipeline_specs.py
"""
from pathlib import Path
from datetime import datetime
from src.registry.pipeline_registry import get_registry

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
    ("Outputs", lambda m: ";".join(f"{o.format}:{'/'.join(o.types)}" for o in m.outputs)),
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
