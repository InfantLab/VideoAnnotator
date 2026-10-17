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


def guide_section(m) -> list[str]:
    """One pipeline's guide (spec 026): what it does, what to set, where it comes from."""
    lines = [f"### {m.guide_title} (`{m.name}`)", "", m.guide_summary, ""]
    behavioural = [
        (k, f) for k, f in m.config_schema.items() if f.group == "behavioural"
    ]
    if behavioural:
        lines.append(
            "What to measure: "
            + ", ".join(
                f"{f.label or k} (`{k}`, default `{f.default}`{' ' + f.unit if f.unit else ''})"
                for k, f in behavioural
            )
            + "."
        )
        lines.append("")
    if m.homepage:
        lines += [f"Homepage: <{m.homepage}>", ""]
    if m.benchmarks:
        for b in m.benchmarks:
            note = f" ({b.note})" if b.note else ""
            lines.append(
                f"- {b.name}: {b.measure} {b.figure}{note}. [Source]({b.source})"
            )
        lines.append("")
    elif m.benchmarks == []:
        lines += ["No published benchmark.", ""]
    if m.quality_note:
        lines += [m.quality_note, ""]
    if m.citations:
        lines += ["Cite: " + ", ".join(f"`{c.key}`" for c in m.citations) + ".", ""]
    return lines


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
        "",
        "## Pipeline guides",
        "",
        *[line for m in metas for line in guide_section(m)],
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    generate()
    print(f"[OK] Wrote {OUTPUT_PATH}")
