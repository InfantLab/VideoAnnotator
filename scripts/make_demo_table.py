"""Regenerate the demo clip's exports from the committed outputs (spec 025).

Writes:
- examples/notebooks/demo_clip_table.csv: the results table the notebooks read
- tests/fixtures/viewer_contract/exported/demo_clip.eaf: the ELAN export
- tests/fixtures/viewer_contract/exported/demo_clip_edited_in_elan.eaf: the same
  with a hand-coded tier added the way ELAN saves one, for the viewer's
  contract test (an exported file must still open after a coder edits it)

Run after the fixtures in tests/fixtures/viewer_contract/ change:
    python scripts/make_demo_table.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "viewer_contract"
TABLE = REPO / "examples" / "notebooks" / "demo_clip_table.csv"
EXPORTED = FIXTURES / "exported"
RUN_NAME = "Peekaboo demo (2026-10-04)"


def demo_run(root: Path) -> Path:
    """The fixtures laid out as a one-video run, as a real run's folder is."""
    run = root / RUN_NAME
    video = run / "demo_clip"
    video.mkdir(parents=True)
    for path in FIXTURES.iterdir():
        if path.is_file() and path.name.startswith("demo_clip_"):
            shutil.copy2(path, video / path.name)
    record = {
        "format": "videoannotator-run",
        "format_version": 1,
        "run": {"batch_id": None, "name": "Peekaboo demo"},
        "videos": [
            {"job_id": None, "folder": "demo_clip", "source": {}, "status": "completed"}
        ],
    }
    (run / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run


def add_coder_tier(source: Path, dest: Path) -> None:
    """`source` with a tier a researcher coded in ELAN, saved as ELAN saves it."""
    from videoannotator.exporters.elan import write_tree

    tree = ET.parse(source)
    root = tree.getroot()
    slots = root.find("TIME_ORDER")
    assert slots is not None
    count = len(slots)
    for i, value in enumerate((1200, 2600, 5000, 7400), start=count + 1):
        ET.SubElement(
            slots, "TIME_SLOT", {"TIME_SLOT_ID": f"ts{i}", "TIME_VALUE": str(value)}
        )
    tier = ET.Element(
        "TIER",
        {
            "ANNOTATOR": "coder1",
            "LINGUISTIC_TYPE_REF": "default-lt",
            "PARTICIPANT": "infant",
            "TIER_ID": "Infant gaze",
        },
    )
    for number, (first, second, value) in enumerate(
        ((count + 1, count + 2, "to mother"), (count + 3, count + 4, "away")), start=900
    ):
        alignable = ET.SubElement(
            ET.SubElement(tier, "ANNOTATION"),
            "ALIGNABLE_ANNOTATION",
            {
                "ANNOTATION_ID": f"a{number}",
                "TIME_SLOT_REF1": f"ts{first}",
                "TIME_SLOT_REF2": f"ts{second}",
            },
        )
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = value
    children = list(root)
    last_tier = max(i for i, child in enumerate(children) if child.tag == "TIER")
    root.insert(last_tier + 1, tier)
    children = list(root)
    last_type = max(i for i, c in enumerate(children) if c.tag == "LINGUISTIC_TYPE")
    root.insert(
        last_type + 1,
        ET.Element(
            "LINGUISTIC_TYPE",
            {
                "GRAPHIC_REFERENCES": "false",
                "LINGUISTIC_TYPE_ID": "default-lt",
                "TIME_ALIGNABLE": "true",
            },
        ),
    )
    ET.SubElement(root, "LOCALE", {"COUNTRY_CODE": "GB", "LANGUAGE_CODE": "en"})
    for stereotype, description in (
        ("Time_Subdivision", "Time subdivision of parent annotation's time interval"),
        ("Symbolic_Subdivision", "Symbolic subdivision of a parent annotation."),
        ("Symbolic_Association", "1-1 association with a parent annotation"),
        (
            "Included_In",
            "Time alignable annotations within the parent annotation's interval",
        ),
    ):
        ET.SubElement(
            root, "CONSTRAINT", {"DESCRIPTION": description, "STEREOTYPE": stereotype}
        )
    ET.indent(tree)
    write_tree(tree, dest)


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from videoannotator.exporters.elan import write_videos
    from videoannotator.exporters.results_set import from_folder
    from videoannotator.exporters.table import write_csv

    with tempfile.TemporaryDirectory() as tmp:
        run = demo_run(Path(tmp))
        results = from_folder(run)
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        with open(TABLE, "w", encoding="utf-8", newline="") as out:
            summary = write_csv(results, out)
        print(f"{TABLE.relative_to(REPO)}: {summary.rows} rows")

        EXPORTED.mkdir(exist_ok=True)
        eaf = EXPORTED / "demo_clip.eaf"
        write_videos(from_folder(run), lambda _video: eaf)
        add_coder_tier(eaf, EXPORTED / "demo_clip_edited_in_elan.eaf")
        print(f"{eaf.relative_to(REPO)} and its ELAN-edited copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
