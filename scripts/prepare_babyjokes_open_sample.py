"""Build the upload folder for the BabyJokes open sample on Zenodo.

    python scripts/prepare_babyjokes_open_sample.py <folder> [--keep-feedback]

Downloads the 54 public clips and their metadata from the old babyjokes repo
(InfantLab/babyjokes, LookitLaughter.test/), checks every row's consent says
public use, and writes into <folder>:

    videos/*.mp4     the clips, names unchanged
    metadata.csv     one row per clip (data dictionary in README.md)
    README.md        the dataset card, also the Zenodo description's source
    SHA256SUMS       for the files above, what `videoannotator demo` checks

Stdlib only, so it runs without extras. ParentFeedback (free text from parents)
is left out unless --keep-feedback; see docs/development/handover_babyjokes_dataset.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://github.com/InfantLab/babyjokes/raw/master/LookitLaughter.test"
SHEET = "_LookitLaughter.test.xlsx"
TREE = "https://api.github.com/repos/InfantLab/babyjokes/git/trees/master?recursive=1"

# Lookit's own consent columns: a clip goes public only if all of these hold.
REQUIRED = {
    "Consent": "accepted",
    "UseOfMedia": "public",
    "Sharing": "public",
    "Withdrawal": "0",
}

COLUMNS = {
    "VideoID": "File name of the clip in videos/.",
    "ChildID": "Lookit's code for the child; the same child across clips.",
    "ParentID": "Lookit's code for the parent.",
    "Age": "Child's age in days on the day of recording.",
    "Gender": "Child's gender as the parent gave it (f, m).",
    "PregnancyLength": "Gestation as the parent gave it.",
    "Languages": "Languages spoken to the child (ISO 639-1), if given.",
    "JokeOrder": "The order the five jokes were shown to this family.",
    "JokeType": "Peekaboo, NomNomNom, ThatsNotAHat, ThatsNotACat or TearingPaper.",
    "JokeNum": "Position of this joke in the family's order (1 to 5).",
    "JokeRep": "Repetition of the joke (1 to 3).",
    "JokeTake": "Recording attempt for that repetition.",
    "HowFunny": "Parent's rating: Not funny, Slightly funny, Funny, Extremely funny.",
    "LaughYesNo": "Parent's report of whether the child laughed (Yes, No).",
    "Child": "Side of the frame the child is on (LEFT, RIGHT).",
    "Parent": "Side of the frame the parent is on (LEFT, RIGHT).",
    "ParentFeedback": "Free-text comment from the parent about the session.",
    "Consent": "Lookit consent outcome.",
    "UseOfMedia": "Parent's choice for use of the recordings (public).",
    "Sharing": "Parent's choice for sharing (public).",
    "DatabraryShare": "Parent agreed to sharing on Databrary (yes, no).",
    "Withdrawal": "1 if the parent later withdrew the recordings.",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return bytes(response.read())


def read_sheet(data: bytes) -> list[dict[str, str]]:
    """The first sheet as rows of strings; openpyxl isn't a core dependency."""
    book = zipfile.ZipFile(io.BytesIO(data))
    shared = [
        re.sub(r"<[^>]+>", "", s)
        for s in re.findall(
            r"<si>(.*?)</si>", book.read("xl/sharedStrings.xml").decode(), re.S
        )
    ]
    sheet = book.read("xl/worksheets/sheet1.xml").decode()
    cell = re.compile(
        r'<c r="([A-Z]+)\d+"([^>]*)>(?:<f>[^<]*</f>)?<v>([^<]*)</v>', re.S
    )
    grid = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
        grid.append(
            {
                col: shared[int(v)] if 't="s"' in attrs else v
                for col, attrs, v in cell.findall(row)
            }
        )
    header, *body = grid
    return [{name: row.get(col, "") for col, name in header.items()} for row in body]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card(rows: list[dict[str, str]], columns: list[str]) -> str:
    children = sorted({r["ChildID"] for r in rows})
    ages = sorted({int(r["Age"]) for r in rows if r["Age"].isdigit()})
    lines = [
        "# BabyJokes: open sample",
        "",
        f"{len(rows)} short videos of {len(children)} parents showing simple jokes to "
        f"their babies (aged {round(ages[0] / 30.44)} to {round(ages[-1] / 30.44)} months), recorded at "
        "home through Lookit, with the parents' ratings of how funny each joke was "
        "and whether the baby laughed.",
        "",
        "Each family saw five jokes (Peekaboo, NomNomNom, ThatsNotAHat, ThatsNotACat, "
        "TearingPaper), each repeated up to three times. File names read "
        "`<child>.joke<position>.rep<repetition>.take<take>.<joke>_h265.mp4`.",
        "",
        "Every parent chose public use of the recordings in Lookit's consent. The "
        "full corpus (about 1,400 videos) is shared with researchers on Databrary.",
        "",
        "## Files",
        "",
        "- `videos/`: the clips (H.265 in MP4).",
        "- `metadata.csv`: one row per clip.",
        "- `SHA256SUMS`: checksums for both.",
        "",
        "## metadata.csv",
        "",
        "| Column | Meaning |",
        "|---|---|",
        *(f"| `{c}` | {COLUMNS.get(c, '')} |" for c in columns),
        "",
        "## Tools",
        "",
        "[VideoAnnotator](https://github.com/InfantLab/VideoAnnotator) uses this sample "
        "for its demo and tutorial.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--keep-feedback", action="store_true")
    args = parser.parse_args()
    out: Path = args.folder
    (out / "videos").mkdir(parents=True, exist_ok=True)

    rows = read_sheet(fetch(f"{SOURCE}/{SHEET}"))
    refused = [
        r["VideoID"]
        for r in rows
        if any(r.get(k, "").strip() != v for k, v in REQUIRED.items())
    ]
    if refused:
        print(f"Not public, left out: {', '.join(refused)}", file=sys.stderr)
    rows = [r for r in rows if r["VideoID"] not in refused]

    listed = {
        Path(t["path"]).name
        for t in json.loads(fetch(TREE))["tree"]
        if t["path"].startswith("LookitLaughter.test/") and t["path"].endswith(".mp4")
    }
    missing = [r["VideoID"] for r in rows if r["VideoID"] not in listed]
    if missing:
        print(f"In the sheet but no video: {', '.join(missing)}", file=sys.stderr)
        rows = [r for r in rows if r["VideoID"] not in missing]
    unlisted = sorted(listed - {r["VideoID"] for r in rows})
    if unlisted:
        print(f"Video but no public row, left out: {', '.join(unlisted)}")

    for row in rows:
        target = out / "videos" / row["VideoID"]
        if not target.exists():
            target.write_bytes(fetch(f"{SOURCE}/{row['VideoID']}"))

    columns = [c for c in COLUMNS if any(c in r for r in rows)]
    if not args.keep_feedback:
        columns.remove("ParentFeedback")
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["VideoID"]))
    (out / "README.md").write_text(card(rows, columns), encoding="utf-8")

    files = [out / "metadata.csv", *sorted((out / "videos").glob("*.mp4"))]
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(out).as_posix()}\n" for p in files)
    )
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(rows)} clips, {size:.0f} MB, in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
